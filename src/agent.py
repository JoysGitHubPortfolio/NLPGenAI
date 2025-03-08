import json

class MakeMyAgent:
    def __init__(self, model, client, model_params, role, response_format, user_input):
        self.model = model 
        self.client = client  
        self.model_params = model_params  
        self.role = role  
        self.response_format = response_format
        self.user_input = user_input  

    def run(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.role},
                {"role": "user", "content": self.user_input}
            ],
            **self.model_params,  
            response_format=self.response_format
        )
        return response.choices[0].message.content 