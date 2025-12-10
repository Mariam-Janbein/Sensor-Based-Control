from TelloInterface_Fixed import TelloInterface
import time

interface = TelloInterface(enable_logging=True, debug=False)  # debug=False

if interface.connect():
    print("Connected successfully!")
else:
    print("Connection failed. Check WiFi/drone power.")
    exit()

if interface.start_trajectory():
    print("Takeoff successful, trajectory started!")
else:
    print("Takeoff failed.")
    interface.disconnect()
    exit()

end_time = time.time() + 30
try:
    while time.time() < end_time:
        interface.step_trajectory()
        time.sleep(0.005)  # Reduced sleep
except KeyboardInterrupt:
    print("Interrupted! Landing and plotting partial logs...")
finally:
    interface.land()
    if interface.logs:
        print("Last log entry:", interface.logs[-1])  # Print last log
    interface.plot_logs()
    interface.disconnect()
    print("Flight complete, logs plotted and saved.")