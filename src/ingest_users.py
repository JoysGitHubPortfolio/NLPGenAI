import pandas as pd
import re

# For regex analysis later on
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

def find_agent(text):
    match = re.match(r'([^:]+):', text)
    return match.group(1) if match else None

# Prepare new data
transcript_ids = []
member_ids = []
claim_ids = []
agent_types = []
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
        agent_type = find_agent(agent_body)
        clean_agent_body = agent_body.replace(agent_type + ": ", "")
        
        member_body = [line.strip().removeprefix('Member: ').removeprefix('Hi, ').lower() 
                    for line in text if 'Member:' in line]
        member_body = ' '.join(member_body)

        # Use regex for analysis
        memeber_id = find_attribute(member_body=member_body, agent_body=agent_body, string=r'mem\d+')
        claim_id = find_attribute(member_body=member_body, agent_body=agent_body, string=r'clm\d+')

        # Output rows
        transcript_ids.append(i)
        member_ids.append(memeber_id)
        claim_ids.append(claim_id)
        agent_types.append(agent_type)
        agent_bodies.append(clean_agent_body)
        member_bodies.append(member_body)
    except:
        print('Error at: ', i)

# Create a DataFrame for new data
df = pd.DataFrame({
    'transcript_id': transcript_ids,
    'member_id': member_ids,
    'claim_id': claim_ids,
    'agent_type': agent_types,
    'agent_body': agent_bodies,
    'member_body': member_bodies,
    'ground_positive': [None] * len(transcript_ids),
    'ground_neutral': [None] * len(transcript_ids),
    'ground_negative': [None] * len(transcript_ids)
})

# Save to CSV
output_file = '../output/transcripts_data.csv'
df.to_csv(output_file, index=False)
print(df.head())
