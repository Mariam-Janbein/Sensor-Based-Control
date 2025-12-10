import numpy as np
import time
from djitellopy import Tello
import logging
from typing import Optional, Dict


class TelloInterface:
    """Interface between controller and Tello hardware"""

    def __init__(self, enable_logging: bool = True, debug: bool = True):  # Changed debug default to True
        self.tello = Tello()
        self.connected = False
        self.in_flight = False
        self.debug = debug

        # State vector: [x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]
        self.state = np.zeros(12)

        # Position tracking
        self.position = np.array([0.0, 0.0, 0.0])
        self.takeoff_position = None

        # Previous state
        self.prev_attitude = np.zeros(3)
        self.prev_height = 0.0
        self.last_update_time = 0.0

        # Velocity tracking
        self.vel_history = []
        self.vel_history_size = 3
        
        # Store last commanded velocities
        self.last_cmd_vel = np.array([0.0, 0.0, 0.0])

        # Command limits
        self.MAX_VEL_CMD = 100

        # Logging
        if enable_logging:
            logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger('TelloInterface')
        else:
            self.logger = None

        # Debug counters
        self.iteration = 0

        # Controller parameters
        self.g = 9.81
        self.m = 0.087
        self.Ix = 2.5e-4
        self.Iy = 2.5e-4
        self.Iz = 4.6e-4
        self.Jr = 5.06e-7
        self.rho_y = 5.0e-4
        self.rho_z = 5.0e-3
        self.d = 0.05

        self.K1_base = 1.0e-5
        self.K2_base = 1.0e-5
        self.K3_base = 1.5e-5
        self.K4_base = 1.06e-4
        self.K5_base = 1.06e-4
        self.K6_base = 1.26e-4
        self.noise_amp = 0.0

        self.M1_base = (self.Iy - self.Iz) / self.Ix
        self.M4_base = (self.Iz - self.Ix) / self.Iy
        self.M7_base = (self.Ix - self.Iy) / self.Iz
        self.M3_base = -self.K1_base / self.Ix
        self.M6_base = -self.K2_base / self.Iy
        self.M8_base = -self.K3_base / self.Iz
        self.M9_base = -self.K4_base / self.m
        self.M10_base = -self.K5_base / self.m
        self.M11_base = -self.K6_base / self.m

        self.N1 = 1 / self.Ix
        self.N2 = 1 / self.Iy
        self.N3 = 1 / self.Iz

        # Increased lambda for faster response
        self.lamb1 = 2.5  # Was 1.5
        self.lamb2 = 2.5  # Was 1.5
        self.m_mu = 0.9

        # Tuned position gains
        self.beta_pos = 2.0  # Was 1.0
        self.gamma_pos = 1.1
        self.k1_pos = 8.0
        self.k2_pos = 6.0

        self.beta_att = 102.15
        self.gamma_att = 1.9
        self.k1_att = 817.6194
        self.k2_att = 2.6997

        self.eps_pos = 0.05
        self.eps_att = 1.0

        # Controller integrals and timing
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_z = 0.0
        self.int_phi = 0.0
        self.int_theta = 0.0
        self.int_psi = 0.0
        self.start_t = None
        self.last_controller_time = None

        self.max_tilt_angle_horizontal = np.deg2rad(25)
        self.max_speed_horizontal = 8.0
        self.max_speed_vertical = 3.0

        # Logs for visualization
        self.logs = []

        self.dead_reckoning_mode = 'hybrid'  # Try hybrid first

    def connect(self) -> bool:
        """Connect to Tello"""
        try:
            if self.logger:
                self.logger.info("Connecting to Tello...")

            self.tello.connect()
            time.sleep(2)

            battery = self.tello.get_battery()
            if self.logger:
                self.logger.info(f"Connected! Battery: {battery}%")

            if battery < 30:
                if self.logger:
                    self.logger.error(f"Battery too low: {battery}%")
                return False

            # Enable mission pads if available
            try:
                self.tello.enable_mission_pads()
                self.tello.set_mission_pad_detection_direction(0)
                if self.logger:
                    self.logger.info("Mission pads enabled")
            except:
                pass

            self.connected = True
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Connection failed: {e}")
            return False

    def takeoff(self) -> bool:
        """Takeoff and initialize position"""
        if not self.connected:
            return False

        try:
            if self.logger:
                self.logger.info("Taking off...")

            self.tello.takeoff()
            time.sleep(4)

            initial_height = self.tello.get_height() / 100.0

            self.position = np.array([0.0, 0.0, initial_height])
            self.takeoff_position = self.position.copy()

            self.state[0:3] = self.position
            self.state[2] = initial_height

            self.last_update_time = time.time()
            self.in_flight = True

            if self.logger:
                self.logger.info(f"Takeoff complete at height: {initial_height:.2f}m")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Takeoff failed: {e}")
            return False

    def land(self) -> bool:
        """Land the drone"""
        try:
            if self.logger:
                self.logger.info("Landing...")

            self.tello.land()
            time.sleep(3)
            self.in_flight = False

            self.position = np.array([0.0, 0.0, 0.0])

            if self.logger:
                self.logger.info("Landed successfully")
            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Landing failed: {e}")
            return False

    def emergency_stop(self):
        """Emergency stop"""
        if self.logger:
            self.logger.error("EMERGENCY STOP!")
        try:
            self.tello.emergency()
        except:
            pass
        self.in_flight = False

    def _get_tello_measurements(self) -> Optional[Dict]:
        """Get all measurements from Tello sensors"""
        try:
            measurements = {
                'height': self.tello.get_height() / 100.0,
                'roll': np.deg2rad(self.tello.get_roll()),
                'pitch': np.deg2rad(self.tello.get_pitch()),
                'yaw': np.deg2rad(self.tello.get_yaw()),
                'speed_x': self.tello.get_speed_x() / 100.0,
                'speed_y': self.tello.get_speed_y() / 100.0,
                'speed_z': self.tello.get_speed_z() / 100.0,
                'battery': self.tello.get_battery(),
                'flight_time': self.tello.get_flight_time()
            }
            return measurements
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Measurement error: {e}")
            return None

    def get_state(self) -> np.ndarray:
        """Get current state estimate"""
        current_time = time.time()
        dt = current_time - self.last_update_time

        meas = self._get_tello_measurements()
        if meas is None:
            return self.state.copy()

        height = meas['height']
        roll = meas['roll']
        pitch = meas['pitch']
        yaw = meas['yaw']
        vx_body = meas['speed_x']
        vy_body = meas['speed_y']
        vz_body = meas['speed_z']

        # Transform velocities from body to world frame
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)

        vx_world = vx_body * c_yaw - vy_body * s_yaw
        vy_world = vx_body * s_yaw + vy_body * c_yaw
        vz_world = vz_body

        # Check if sensor velocities are near zero (might be unreliable)
        sensor_vel_mag = np.sqrt(vx_world**2 + vy_world**2)
        cmd_vel_mag = np.sqrt(self.last_cmd_vel[0]**2 + self.last_cmd_vel[1]**2)
        
        # Hybrid dead reckoning: use commanded velocities if sensor readings are suspiciously low
        if self.dead_reckoning_mode == 'hybrid':
            if sensor_vel_mag < 0.05 and cmd_vel_mag > 0.2:
                # Sensor readings likely unreliable, use commanded velocities
                vx_world = self.last_cmd_vel[0]
                vy_world = self.last_cmd_vel[1]
                if self.debug and self.iteration % 50 == 0:
                    if self.logger:
                        self.logger.debug(f"Using commanded velocities for dead reckoning")

        # Minimal filtering
        current_vel = np.array([vx_world, vy_world, vz_world])
        self.vel_history.append(current_vel)
        if len(self.vel_history) > self.vel_history_size:
            self.vel_history.pop(0)
        vel_filtered = np.mean(self.vel_history, axis=0) if self.vel_history else current_vel

        # Update position using dead reckoning
        if 0 < dt < 1.0:
            self.position[0] += vel_filtered[0] * dt
            self.position[1] += vel_filtered[1] * dt
            self.position[2] = height

        # Mission pad correction if detected
        try:
            pad_id = self.tello.get_mission_pad_id()
            if pad_id != -1:
                pad_x = self.tello.get_mission_pad_distance_x() / 100.0
                pad_y = self.tello.get_mission_pad_distance_y() / 100.0
                pad_z = self.tello.get_mission_pad_distance_z() / 100.0
                
                # Blend with current estimate
                self.position[0] = 0.5 * self.position[0] + 0.5 * pad_x
                self.position[1] = 0.5 * self.position[1] + 0.5 * pad_y
                self.position[2] = 0.5 * self.position[2] + 0.5 * pad_z
                
                if self.logger and self.iteration % 50 == 0:
                    self.logger.info(f"Mission pad {pad_id} detected at [{pad_x:.2f}, {pad_y:.2f}, {pad_z:.2f}]")
        except:
            pass

        # Estimate angular rates
        if dt > 0 and dt < 1.0:
            p_est = (roll - self.prev_attitude[0]) / dt
            q_est = (pitch - self.prev_attitude[1]) / dt
            yaw_diff = yaw - self.prev_attitude[2]
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            r_est = yaw_diff / dt
        else:
            p_est, q_est, r_est = 0.0, 0.0, 0.0

        # Construct state vector
        self.state[0:3] = self.position
        self.state[3:6] = vel_filtered
        self.state[6:9] = [roll, pitch, yaw]
        self.state[9:12] = [p_est, q_est, r_est]

        # Enhanced debug output
        if self.debug and self.iteration % 10 == 0:
            if self.logger:
                self.logger.debug(
                    f"[{self.iteration:4d}] "
                    f"Pos: [{self.position[0]:6.3f}, {self.position[1]:6.3f}, {self.position[2]:6.3f}] | "
                    f"Vel(sensor): [{vx_body:5.2f}, {vy_body:5.2f}] → [{vx_world:5.2f}, {vy_world:5.2f}] | "
                    f"Vel(cmd): [{self.last_cmd_vel[0]:5.2f}, {self.last_cmd_vel[1]:5.2f}]"
                )

        # Update history
        self.prev_attitude = np.array([roll, pitch, yaw])
        self.prev_height = height
        self.last_update_time = current_time
        self.iteration += 1

        return self.state.copy()

    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """Send velocity command to Tello"""
        if not self.in_flight:
            return

        # Store commanded velocities for dead reckoning
        self.last_cmd_vel = np.array([vx, vy, vz])

        # Transform to body frame
        yaw = self.state[8]
        c_yaw = np.cos(yaw)
        s_yaw = np.sin(yaw)

        vx_body = vx * c_yaw + vy * s_yaw
        vy_body = -vx * s_yaw + vy * c_yaw
        vz_body = vz

        # Convert to Tello units (cm/s) - NOTE: Tello SDK max is 100
        vx_cmd = int(np.clip(vx_body * 100, -self.MAX_VEL_CMD, self.MAX_VEL_CMD))
        vy_cmd = int(np.clip(vy_body * 100, -self.MAX_VEL_CMD, self.MAX_VEL_CMD))
        vz_cmd = int(np.clip(vz_body * 100, -self.MAX_VEL_CMD, self.MAX_VEL_CMD))
        yaw_rate_cmd = int(np.clip(np.rad2deg(yaw_rate), -100, 100))

        # Enhanced debug output
        if self.debug and self.iteration % 10 == 0:
            if self.logger:
                self.logger.debug(
                    f"CMD: World=[{vx:6.3f}, {vy:6.3f}, {vz:6.3f}] m/s → "
                    f"Body=[{vx_body:6.3f}, {vy_body:6.3f}, {vz_body:6.3f}] → "
                    f"RC=[{vx_cmd:4d}, {vy_cmd:4d}, {vz_cmd:4d}] cm/s, yaw={yaw_rate_cmd:4d}°/s"
                )

        try:
            # Tello RC: (left_right, forward_backward, up_down, yaw)
            # Body frame: vx=forward, vy=right
            self.tello.send_rc_control(vy_cmd, vx_cmd, vz_cmd, yaw_rate_cmd)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Command send failed: {e}")

    def get_position(self) -> np.ndarray:
        """Get current position"""
        return self.state[0:3].copy()

    def get_attitude(self) -> np.ndarray:
        """Get current attitude"""
        return self.state[6:9].copy()

    def get_battery(self) -> int:
        """Get battery percentage"""
        try:
            return self.tello.get_battery()
        except:
            return -1

    def disconnect(self):
        """Disconnect from Tello"""
        if self.in_flight:
            self.land()

        try:
            try:
                self.tello.disable_mission_pads()
            except:
                pass
            self.tello.end()
            if self.logger:
                self.logger.info("Disconnected from Tello")
        except:
            pass

        self.connected = False

    # Controller methods

    def desired_pos(self, t):
        xdes = np.sin(0.5 * t)
        dxdes = 0.5 * np.cos(0.5 * t)
        ddxdes = -0.25 * np.sin(0.5 * t)
        ydes = np.cos(0.5 * t)
        dydes = -0.5 * np.sin(0.5 * t)
        ddydes = -0.25 * np.cos(0.5 * t)
        zdes = min(0.2 * t + 0.5, 1.5)
        dzdes = 0.2 if zdes < 1.5 else 0.0
        ddzdes = 0.0
        return xdes, dxdes, ddxdes, ydes, dydes, ddydes, zdes, dzdes, ddzdes

    def desired_psi(self, t):
        psides = 0.3
        dpsides = 0.0
        ddpsides = 0.0
        return psides, dpsides, ddpsides

    def start_trajectory(self) -> bool:
        """Start the trajectory tracking controller"""
        if not self.takeoff():
            return False

        self.start_t = time.time()
        self.last_controller_time = self.start_t
        self.int_x = 0.0
        self.int_y = 0.0
        self.int_z = 0.0
        self.int_psi = 0.0

        if self.logger:
            self.logger.info("Trajectory controller started")
        return True

    def step_trajectory(self):
        """Execute one step of the controller"""
        if not self.in_flight:
            return

        current_time = time.time()
        t = current_time - self.start_t
        dt = current_time - self.last_controller_time
        if dt <= 0:
            return

        # Get desired values
        xdes, dxdes, ddxdes, ydes, dydes, ddydes, zdes, dzdes, ddzdes = self.desired_pos(t)
        psides, dpsides, ddpsides = self.desired_psi(t)

        # Get current state
        state = self.get_state()
        x, y, z = state[0], state[1], state[2]
        dx, dy, dz = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]
        dphi, dtheta, dpsi = state[9], state[10], state[11]

        # Position errors - NO ARTIFICIAL DISTURBANCES
        ex = x - xdes
        ey = y - ydes
        ez = z - zdes
        dex = dx - dxdes
        dey = dy - dydes
        dez = dz - dzdes

        # Integral updates
        d_int_x = np.abs(ex)**self.m_mu * np.sign(ex)
        self.int_x += d_int_x * dt
        d_int_y = np.abs(ey)**self.m_mu * np.sign(ey)
        self.int_y += d_int_y * dt
        d_int_z = np.abs(ez)**self.m_mu * np.sign(ez)
        self.int_z += d_int_z * dt

        # Position ITSM
        s7 = self.lamb1 * ex + self.lamb2 * self.int_x
        ds7 = self.lamb1 * dex + self.lamb2 * d_int_x
        s9 = self.lamb1 * ey + self.lamb2 * self.int_y
        ds9 = self.lamb1 * dey + self.lamb2 * d_int_y
        s11 = self.lamb1 * ez + self.lamb2 * self.int_z
        ds11 = self.lamb1 * dez + self.lamb2 * d_int_z

        # Clip ds
        abs_ds7 = np.clip(np.abs(ds7), 1e-10, 1e6)
        abs_ds9 = np.clip(np.abs(ds9), 1e-10, 1e6)
        abs_ds11 = np.clip(np.abs(ds11), 1e-10, 1e6)

        # Smooth sign
        sign_ds7 = ds7 / (abs_ds7 + self.eps_pos)
        sign_ds9 = ds9 / (abs_ds9 + self.eps_pos)
        sign_ds11 = ds11 / (abs_ds11 + self.eps_pos)

        # Hyperplane variables
        sigma7 = s7 + (1 / self.beta_pos) * sign_ds7 * (abs_ds7 ** self.gamma_pos)
        sigma9 = s9 + (1 / self.beta_pos) * sign_ds9 * (abs_ds9 ** self.gamma_pos)
        sigma11 = s11 + (1 / self.beta_pos) * sign_ds11 * (abs_ds11 ** self.gamma_pos)

        # Smooth sign for sigma
        sign_sigma7 = sigma7 / (np.abs(sigma7) + self.eps_pos)
        sign_sigma9 = sigma9 / (np.abs(sigma9) + self.eps_pos)
        sign_sigma11 = sigma11 / (np.abs(sigma11) + self.eps_pos)

        # Position continuous
        term_x = self.m_mu * self.lamb2**2 / self.lamb1 * np.abs(ex)**(2*self.m_mu - 1) * np.sign(ex)
        vxc = ddxdes - self.M9_base * dx + 1 / self.lamb1 * (term_x - self.beta_pos / self.gamma_pos * sign_ds7 * abs_ds7**(2 - self.gamma_pos))
        term_y = self.m_mu * self.lamb2**2 / self.lamb1 * np.abs(ey)**(2*self.m_mu - 1) * np.sign(ey)
        vyc = ddydes - self.M10_base * dy + 1 / self.lamb1 * (term_y - self.beta_pos / self.gamma_pos * sign_ds9 * abs_ds9**(2 - self.gamma_pos))
        term_z = self.m_mu * self.lamb2**2 / self.lamb1 * np.abs(ez)**(2*self.m_mu - 1) * np.sign(ez)
        vzc = ddzdes - self.M11_base * dz + self.g + 1 / self.lamb1 * (term_z - self.beta_pos / self.gamma_pos * sign_ds11 * abs_ds11**(2 - self.gamma_pos))

        # Position switching
        vxs = 1 / self.lamb1 * (-self.k1_pos * sigma7 - self.k2_pos * sign_sigma7)
        vys = 1 / self.lamb1 * (-self.k1_pos * sigma9 - self.k2_pos * sign_sigma9)
        vzs = 1 / self.lamb1 * (-self.k1_pos * sigma11 - self.k2_pos * sign_sigma11)

        vx = vxc + vxs
        vy = vyc + vys
        vz = vzc + vzs

        # Compute desired velocities
        v_des_x = dx + vx * dt
        v_des_y = dy + vy * dt
        v_des_z = dz + vz * dt

        # Safety limit for Z
        if z > 1.45:
            v_des_z = min(v_des_z, 0)

        # Attitude control for psi
        e5 = psi - psides
        de5 = dpsi - dpsides

        d_int_psi = np.abs(e5)**self.m_mu * np.sign(e5)
        self.int_psi += d_int_psi * dt

        s5 = self.lamb1 * e5 + self.lamb2 * self.int_psi
        ds5 = self.lamb1 * de5 + self.lamb2 * d_int_psi

        abs_ds5 = np.clip(np.abs(ds5), 1e-10, 1e6)
        sign_ds5 = ds5 / (abs_ds5 + self.eps_att)

        sigma5 = s5 + (1 / self.beta_att) * sign_ds5 * (abs_ds5 ** self.gamma_att)
        sign_sigma5 = sigma5 / (np.abs(sigma5) + self.eps_att)

        term_psi = self.m_mu * self.lamb2**2 / self.lamb1 * np.abs(e5)**(2*self.m_mu - 1) * np.sign(e5)
        u4c = 1 / (self.N3 * self.lamb1) * (term_psi - self.beta_att / self.gamma_att * sign_ds5 * abs_ds5**(2 - self.gamma_att) - self.lamb1 * (self.M7_base * dtheta * dpsi + self.M8_base * dpsi**2) + ddpsides)
        u4s = 1 / (self.N3 * self.lamb1) * (-self.k1_att * sigma5 - self.k2_att * sign_sigma5)

        u4 = u4c + u4s
        yaw_rate_des = dpsi + (u4 * self.N3) * dt

        # Send commands
        self.send_velocity_command(v_des_x, v_des_y, v_des_z, yaw_rate_des)

        # Update time
        self.last_controller_time = current_time

        # Log data
        self.logs.append({
            't': t,
            'x': x, 'y': y, 'z': z,
            'dx': dx, 'dy': dy, 'dz': dz,
            'xdes': xdes, 'ydes': ydes, 'zdes': zdes,
            'phi': phi, 'theta': theta, 'psi': psi,
            'psides': psides,
            'v_des_x': v_des_x, 'v_des_y': v_des_y, 'v_des_z': v_des_z,
            'ex': ex, 'ey': ey, 'ez': ez
        })

        # Enhanced debug output
        if self.debug and self.iteration % 10 == 0:
            if self.logger:
                self.logger.debug(
                    f"t={t:.2f}s | "
                    f"Pos=[{x:6.3f}, {y:6.3f}, {z:6.3f}] | "
                    f"Des=[{xdes:6.3f}, {ydes:6.3f}, {zdes:6.3f}] | "
                    f"Err=[{ex:6.3f}, {ey:6.3f}, {ez:6.3f}] | "
                    f"Cmd=[{v_des_x:5.2f}, {v_des_y:5.2f}, {v_des_z:5.2f}] m/s"
                )

    def plot_logs(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import pandas as pd

        if not self.logs:
            print("No logs to plot.")
            return

        # Save CSV
        df = pd.DataFrame(self.logs)
        df.to_csv('flight_logs.csv', index=False)
        print("Logs saved to flight_logs.csv")

        t = [log['t'] for log in self.logs]
        x = [log['x'] for log in self.logs]
        y = [log['y'] for log in self.logs]
        z = [log['z'] for log in self.logs]
        xdes = [log['xdes'] for log in self.logs]
        ydes = [log['ydes'] for log in self.logs]
        zdes = [log['zdes'] for log in self.logs]
        ex = [log['ex'] for log in self.logs]
        ey = [log['ey'] for log in self.logs]
        ez = [log['ez'] for log in self.logs]

        # Position tracking plot
        plt.figure(figsize=(12, 10))
        plt.subplot(3, 1, 1)
        plt.plot(t, x, 'b', linewidth=2, label='x actual')
        plt.plot(t, xdes, 'r--', linewidth=2, label='x desired')
        plt.ylabel('X Position (m)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(t, y, 'b', linewidth=2, label='y actual')
        plt.plot(t, ydes, 'r--', linewidth=2, label='y desired')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(t, z, 'b', linewidth=2, label='z actual')
        plt.plot(t, zdes, 'r--', linewidth=2, label='z desired')
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle('Position Tracking', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('position_tracking.png', dpi=150)
        plt.show()

        # Tracking errors
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, ex, 'r', linewidth=2)
        plt.ylabel('X Error (m)')
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(t, ey, 'r', linewidth=2)
        plt.ylabel('Y Error (m)')
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(t, ez, 'r', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Z Error (m)')
        plt.grid(True)
        plt.suptitle('Tracking Errors', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('tracking_errors.png', dpi=150)
        plt.show()

        # 2D path plot
        plt.figure(figsize=(10, 10))
        plt.plot(x, y, 'b', linewidth=3, label='Actual', marker='o', markersize=3, markevery=10)
        plt.plot(xdes, ydes, 'r--', linewidth=2, label='Desired')
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.axis('equal')
        plt.title('2D Path Following', fontsize=14, fontweight='bold')
        plt.savefig('2d_path.png', dpi=150)
        plt.show()

        # 3D path plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b', linewidth=3, label='Actual')
        ax.plot(xdes, ydes, zdes, 'r--', linewidth=2, label='Desired')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        ax.legend(fontsize=12)
        ax.set_title('3D Path Following', fontsize=14, fontweight='bold')
        plt.savefig('3d_path.png', dpi=150)
        plt.show()

        print("All plots saved and displayed.")
