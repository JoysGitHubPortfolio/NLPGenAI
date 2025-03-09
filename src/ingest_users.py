import pandas as pd
import re

def find_attribute(member_body, agent_body, string):
    agent_search = re.search(string, member_body)
    member_search = re.search(string, agent_body)
    if agent_search:
        found_attribute = agent_search.group()
    elif member_search:
        found_attribute = member_search.group()
    else:
        found_attribute = None
    return found_attribute

verbose = False
transcript_ids = []
member_ids = []
claim_ids = []
agent_bodies = []
member_bodies = []
for i in range(0, 200):
    try:
        with open(f"../transcripts/transcript_{i}.txt") as f:
            text = f.readlines()

        # Filter by person
        agent_body = [line.strip().lower() for line in text
                      if 'Support:' in line
                      or 'Agent:' in line]
        agent_body = ' '.join(agent_body)
        
        member_body = [line.strip().removeprefix('Member: Hi, ').lower() 
                    for line in text if 'Member:' in line]
        member_body = ' '.join(member_body)

        # Use regex for analysis
        memeber_id = find_attribute(member_body=member_body, agent_body=agent_body, string=r'mem\d+')
        claim_id = find_attribute(member_body=member_body, agent_body=agent_body, string=r'clm\d+')

        # Output rows
        transcript_ids.append(i)
        member_ids.append(memeber_id)
        claim_ids.append(claim_id)
        agent_bodies.append(agent_body)
        member_bodies.append(member_body)
    except:
        print('Error at: ', i)


df = pd.DataFrame({
    'transcript_id': transcript_ids,
    'member_id': member_ids,
    'claim_id': claim_ids,
    'agent_body': agent_bodies,
    'member_body': member_bodies
})

output_file = '../output/transcripts_data.csv'
df.to_csv(output_file, index=False)
print(df.head())  # Print the first 5 rows for a quick preview