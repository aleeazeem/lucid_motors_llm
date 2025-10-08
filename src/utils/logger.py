from typing import Dict, Any

class Logger:
    
    def __init__(self):
        pass
    
    def log_pipeline(self, message: str) -> None:
        print()
        message = f"************************ {message} *************************"
        print("*"*len(message))
        print(message)
        print("*"*len(message))
        print()
    
    def log_step(self, message: str) -> None:
        print()
        message = f"========== {message} =========="
        print("="*len(message))
        print(message)
        print("="*len(message))

    def info(self, message: str) -> None:
        print(f"[LOG]: {message}")
    
    def log_messages(self,messages: list) -> None:
        for msg in messages:
            self.info(msg)

    def error(self, message: str) -> None:
        print(f"[ERROR]: {message}")