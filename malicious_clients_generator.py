import random

# Configuration
total_clients = 300
percent_steps = [50, 40, 30, 20, 10]

LIST_SF_ATTACKERS = []
LIST_LF_ATTACKERS = []

all_clients = list(range(total_clients))

for percent in percent_steps:
    num_clients = total_clients * percent // 100
    num_sf = int(num_clients * 0.7)
    num_lf = num_clients - num_sf

    # Sample clients, ensuring LF > SF
    random.shuffle(all_clients)
    sorted_clients = sorted(all_clients[:num_clients])

    sf_candidates = sorted_clients[:num_sf]
    lf_candidates = sorted_clients[-num_lf:]

    # Ensure LF IDs are greater than SF IDs
    if lf_candidates and sf_candidates and min(lf_candidates) <= max(sf_candidates):
        # Re-pick LF candidates from the top end
        lf_candidates = [id for id in sorted_clients if id > max(sf_candidates)][:num_lf]

    sf_attackers = sorted(sf_candidates)
    lf_attackers = sorted(lf_candidates)

    LIST_SF_ATTACKERS.append(",".join(map(str, sf_attackers)))
    LIST_LF_ATTACKERS.append(",".join(map(str, lf_attackers)))

    print(f"{percent}%:")
    print(f"  SF Attackers ({len(sf_attackers)} clients): {LIST_SF_ATTACKERS[-1]}")
    print(f"  LF Attackers ({len(lf_attackers)} clients): {LIST_LF_ATTACKERS[-1]}")
    print(f"  Total: {len(sf_attackers) + len(lf_attackers)} clients ✅\n")

# Reverse the arrays
LIST_SF_ATTACKERS.reverse()
LIST_LF_ATTACKERS.reverse()

print("\n--- After Reversing Rows ---")
for i, (sf_row, lf_row) in enumerate(zip(LIST_SF_ATTACKERS, LIST_LF_ATTACKERS)):
    reversed_percent = percent_steps[::-1][i]
    sf_len = len(sf_row.split(","))
    lf_len = len(lf_row.split(","))
    print(f"{reversed_percent}%:")
    print(f"  SF Attackers ({sf_len} clients): {sf_row}")
    print(f"  LF Attackers ({lf_len} clients): {lf_row}")
    print(f"  Total: {sf_len + lf_len} clients ✅\n")

# Final declarations
print("\nLIST_SF_ATTACKERS=(")
for row in LIST_SF_ATTACKERS:
    print(f'    "{row}"')
print(")")

print("\nLIST_LF_ATTACKERS=(")
for row in LIST_LF_ATTACKERS:
    print(f'    "{row}"')
print(")")
