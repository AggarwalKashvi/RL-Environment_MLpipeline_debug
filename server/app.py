import uvicorn
from openenv.core.env_server import create_fastapi_app
from env.environment import MLPipelineDebugEnv
from models.action import Action
from models.observation import Observation

# 1. Automatically generate the spec-compliant FastAPI app
app = create_fastapi_app(MLPipelineDebugEnv, Action, Observation)

# 2. Add the main() function required by the validator
def main():
    print("Starting OpenEnv server...")
    uvicorn.run(
        "server.app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )

# 3. Add the callable block required by the validator
if __name__ == "__main__":
    main()