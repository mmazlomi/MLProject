import pandas as pd
import hashlib
import time
import matplotlib.pyplot as plt

input_path = "../dataset/hotel_bookings.csv"

df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)
print("First 5 records:")
print(df.head())

def row_md5(row):
    row_string = "|".join(
        str(value) for value in row
    )

    return hashlib.md5(
        row_string.encode("utf-8")
    ).hexdigest()


start = time.perf_counter()

df["md5_hash"] = df.apply(row_md5, axis=1)

md5_duplicates = df["md5_hash"].duplicated(keep="first")

md5_duplicate_count = md5_duplicates.sum()

md5_time = time.perf_counter() - start

print("MD5 duplicate records:", md5_duplicate_count)
print(f"MD5 execution time: {md5_time:.4f} seconds")


start = time.perf_counter()

pandas_duplicates = df.drop(
    
).duplicated(keep="first")

pandas_duplicate_count = pandas_duplicates.sum()

pandas_time = time.perf_counter() - start

print("Pandas duplicate records:", pandas_duplicate_count)
print(f"Pandas execution time: {pandas_time:.4f} seconds")


print("\n===== DUPLICATE DETECTION COMPARISON =====")

print("MD5 duplicates:    ", md5_duplicate_count)
print("Pandas duplicates: ", pandas_duplicate_count)

print(f"MD5 time:          {md5_time:.4f} seconds")
print(f"Pandas time:       {pandas_time:.4f} seconds")

print("\nResults match:",
      md5_duplicate_count == pandas_duplicate_count)
      
      

start = time.perf_counter()

row_hashes = pd.util.hash_pandas_object(
    df.drop(columns=["md5_hash"]),
    index=False
)

pandas_hash_duplicates = row_hashes.duplicated()

hash_time = time.perf_counter() - start

print("Hash-based duplicates:", pandas_hash_duplicates.sum())
print(f"Hashing time: {hash_time:.4f} seconds")


"""
methods = ["MD5", "Pandas"]
execution_times = [md5_time, pandas_time]

plt.figure(figsize=(8, 5))

plt.bar(methods, execution_times)

plt.ylabel("Execution Time (seconds)")
plt.title("Duplicate Detection Performance")

for i, value in enumerate(execution_times):
    plt.text(i, value, f"{value:.4f}s", ha="center", va="bottom")

plt.show()  
"""    