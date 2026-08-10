import pickle

with open("data_Low_D_space_full_data_BL_sporadic.pkl", "rb") as f:
    data = pickle.load(f)

print(data)

# extract target variables (if present) and save them to a new pickle file
eigen_vals = data.get('eigen_vals') if isinstance(data, dict) else None
exceeding_eig_space = data.get('exceeding_eig_space') if isinstance(data, dict) else None

baseline = {
    'eigen_vals': eigen_vals,
    'exceeding_eig_space': exceeding_eig_space,
}

with open("baseline_space.pkl", "wb") as f:
    pickle.dump(baseline, f)

print("Saved baseline_space.pkl")

