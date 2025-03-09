member_dictionary = {}

for i in range(0, 199):
    with open(f"../transcripts/transcript_{i}.txt") as f:
        text = f.readlines()

    # Filter lines containing "Member"
    member_body = [line.strip().removeprefix('Member: ') 
                for line in text if 'Member: ' in line]
    member_body = ' '.join(member_body)
    member_dictionary[i] = member_body

if __name__ == "__main__":
    for k,v in member_dictionary.items():
        print(k,v)
        print()