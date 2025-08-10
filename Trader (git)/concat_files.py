# concat_files.py

FILE_1 = "ml_dataset.csv"
FILE_2 = "ml_dataset2.csv"
OUTPUT_FILE = "combined_output.csv"
SWITCH_LINE = "-- SWITCH --"

def concat_with_switch_line(file1, file2, output_file, switch_line):
    with open(file1, "r") as f1, open(file2, "r") as f2, open(output_file, "w") as out:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        # Write contents of the first file
        out.writelines(lines1)

        # Write switch line with a newline if not already included
        if not lines1[-1].endswith('\n'):
            out.write('\n')
        out.write(switch_line + '\n')

        # Write contents of the second file
        out.writelines(lines2)

    print(f"Files concatenated into {output_file}")

if __name__ == "__main__":
    concat_with_switch_line(FILE_1, FILE_2, OUTPUT_FILE, SWITCH_LINE)
