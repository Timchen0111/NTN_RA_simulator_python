import numpy as np
import matplotlib.pyplot as plt

# --- 1. Initialization ---
source_number = 150
n_sc = 48  # Number of sub-carriers
packet_bit = 328  # Packet bit length
simulation_time = 1e2

# --- DSA part removed ---
# We are now only running the Slotted ALOHA (SA) case, which is equivalent
# to DSA with N_replica = 1.
n_replica = 1  

# Define the traffic load range
load_range = np.arange(0, 91, 1)  # G = 0, 1, ..., 90

# Lists to store results
packet_loss_ratios = []
throughput_list = []

print(f"Starting simulation...")
print(f"Configuration: source_number={source_number}, n_sc={n_sc}")

# --- 2. Main Simulation Loop ---

# Loop over different traffic loads (G)
for G in load_range:
    # Generate Poisson arrival parameters
    g_bps = G * 1000  # Convert kbps to bps
    n_pk = g_bps / packet_bit  # Average number of packets per T_EDT
    lambda_val = n_pk  # Average arrival rate
    # Probability of transmission for a single source, based on aggregate Poisson model
    pr = 1 - np.exp(-lambda_val / source_number)
    # Initialize counters for this load (G)
    ackd_packet_count = 0
    send_packet_count = 0
    time = 0
    # Run simulation for `simulation_time` slots
    while time < simulation_time:
        time += 1
        # Status for UEs in the current slot (1 = active, 0 = decoded)
        source_status = np.zeros(source_number, dtype=int)
        # Resource grid for this slot, counting attempts
        transmission_attempts = np.zeros((1, n_sc), dtype=int)
        # Resource grid tracking which source transmitted
        # Note: This simple model only tracks the *last* source if multiple transmit
        # on the same resource, but it's fine for collision check (count > 1).
        attempt_source = np.zeros((1, n_sc), dtype=int)
        # --- 2a. Packet Arrival and Transmission ---
        for source in range(source_number):
            if np.random.rand() <= pr:  # New packet arrives
                send_packet_count += 1
                source_status[source] = 1  # Mark as active (needs ACK)
                # --- This is the SA part (n_replica = 1) ---
                # Randomly select N_replica EDT occasions (time)
                # np.random.permutation gives indices from 0 to n_edt-1
                # Randomly select a carrier (frequency)
                # np.random.randint gives index from 0 to n_sc-1
                chosen_carrier = np.random.randint(n_sc)
                # Mark the resource
                transmission_attempts[0, chosen_carrier] += 1
                attempt_source[0, chosen_carrier] = source
        
        # --- 2b. Collision Check and Decoding ---
        for i in range(n_sc):
            # Check for a successful transmission (only 1 packet)
            if transmission_attempts[0, i] == 1:
                current_ue = attempt_source[0, i]
                # Check if this UE's packet hasn't been decoded yet
                if source_status[current_ue] == 1:
                    ackd_packet_count += 1
                    source_status[current_ue] = 0  # Mark as decoded
    
    # --- 3. Calculate and Store Results for this Load (G) ---
    if send_packet_count > 0:
        loss_ratio = 1 - (ackd_packet_count / send_packet_count)
    else:
        loss_ratio = 0.0  # No packets sent, so no loss
        
    packet_loss_ratios.append(loss_ratio)
    
    # Throughput = Offered_Load * Success_Probability
    # Success_Probability = 1 - Packet_Loss_Ratio
    throughput = G * (1 - loss_ratio)
    throughput_list.append(throughput)
    
    if G % 10 == 0:
        print(f"  Processed G = {G} kbps... PLR = {loss_ratio:.4f}, Throughput = {throughput:.2f} kbps")

print("Simulation complete.")

# Convert lists to numpy arrays for plotting
load_array = np.array(load_range)
plr_array = np.array(packet_loss_ratios)
throughput_array = np.array(throughput_list)


# --- 4. Plot Results ---

# Figure 1: Average Throughput
plt.figure(1, figsize=(10, 6))
plt.plot(load_array, throughput_array, '-', color='red', label='SA')
plt.title('Average Throughput')
plt.xlabel('G (Offered Traffic, kbps)')
plt.ylabel('Average Throughput (kbps)')
plt.legend(loc='best')
plt.grid(True)

# Figure 2: Packet Loss Ratio
plt.figure(2, figsize=(10, 6))
plt.plot(load_array, plr_array, '-', color='red', label='SA')
plt.title('Packet Loss Ratio')
plt.xlabel('G (Offered Traffic, kbps)')
plt.ylabel('Packet Loss Ratio')
plt.yscale('log')  # Set y-axis to log scale
plt.legend(loc='best')
plt.grid(True)

# Show plots
plt.show()