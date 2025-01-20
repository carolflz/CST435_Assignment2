from mpi4py import MPI
import csv
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    start_time = MPI.Wtime()  # Start timing

    # Load the data only on rank 0
    if rank == 0:
        with open("daily.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row
            lines = list(reader)

        # Divide data among processes
        chunk_size = len(lines) // size
        chunks = [lines[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]

        # Handle any remaining lines
        if len(lines) % size != 0:
            chunks[-1].extend(lines[size * chunk_size:])
    else:
        chunks = None

    # Scatter chunks of data to all processes
    data_chunk = comm.scatter(chunks, root=0)

    # Each process computes local percent changes
    local_results = []
    try:
        previous_country = None
        previous_fx = None

        for row in data_chunk:
            # Skip rows with missing or invalid data
            if len(row) < 3 or not row[2].strip():
                continue

            try:
                current_country, current_fx = row[1], float(row[2])
            except ValueError:
                continue

            if previous_country is not None and current_country == previous_country:
                percent_change = ((current_fx - previous_fx) / previous_fx) * 100.00
                percent_change = round(percent_change, 2)
                local_results.append((current_country, percent_change))

            previous_country = current_country
            previous_fx = current_fx

    except Exception as e:
        print(f"Exception on rank {rank}: {e}")
        sys.exit(1)

    # Gather all results at rank 0
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Flatten the gathered results
        flat_results = [item for sublist in all_results for item in sublist]

        # Aggregate and count results
        aggregated_results = {}
        for country, percent_change in flat_results:
            key = f"{country}: {percent_change:.2f}%"
            if key in aggregated_results:
                aggregated_results[key] += 1
            else:
                aggregated_results[key] = 1

        # Write results to a file
        with open("output.txt", "w") as output_file:
            for key, count in sorted(aggregated_results.items()):
                output_file.write(f"{key}\t{count}\n")

        end_time = MPI.Wtime()  # End timing
        print(f"Execution Time: {end_time - start_time:.2f} seconds")
