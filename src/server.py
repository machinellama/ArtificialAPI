import os
from dotenv import load_dotenv
from src.init import create_app

load_dotenv()

def main():
  app = create_app()
  host = "0.0.0.0"
  port = 5700
  debug = True

  app.run(host = host, port = port, debug = debug)

if __name__ == "__main__":
  main()
