class key:
    def __init__(self):
        self.d={
            "GEMINI_API_KEY":"",
            "LANGSMITH_API_KEY":"",
            "HUGGINGFACE_API_KEY":""
            }
        
    def get_key(self,str):
        try :
            return self.d[str]
        except:
            return ""
    