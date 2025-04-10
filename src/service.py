from http import HTTPStatus
from datetime import datetime, timedelta

from pydantic import BaseModel

from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt

import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException, InvalidArgument


# ----- USER CREDENTIALS -----

# NOTE This is a simple in-memory user store for demonstration purposes.
# In production, use a secure database

USERS = {
    "alice": "alicepassword",
    "bob": "bobpassword",
    "charlie": "charliepassword",
}

# ----- MIDDLEWARE FOR JWT AUTHENTICATION -----

# NOTE This is a simple JWT authentication middleware for demonstration purposes.
# In production, use a more robust authentication mechanism

JWT_SECRET_KEY = "jwt_secret_key" 
JWT_ALGORITHM = "HS256"

class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response
        
    
# ----- BENTOML SERVICE -----

# init runner & service
regression_model_runner = bentoml.sklearn.get("admission_regression_model:latest").to_runner()

regression_model_service = bentoml.Service("regression_model_service", runners=[regression_model_runner])

# adding middleware to the service
regression_model_service.add_asgi_middleware(JWTAuthMiddleware)

# ---- ENDPOINT : LOGIN ----

# utilities

def create_jwt_token(user_id: str, timedelta_before_exp_sec) -> str:
    """Create a JWT token for the user."""
    return jwt.encode(
        {
            "sub": user_id, 
            "exp": datetime.now() + timedelta(seconds=timedelta_before_exp_sec)
        }, 
        JWT_SECRET_KEY, 
        algorithm=JWT_ALGORITHM
    )

class UnauthorizedException(BentoMLException):
    """Custom exception for unauthorized access"""
    error_code = HTTPStatus.UNAUTHORIZED

class InputLogin(BaseModel):
    """Input model for login endpoint."""
    username: str
    password: str
    timedelta_before_exp_sec: int = 3600

# endpoint
@regression_model_service.api(
        input=JSON(pydantic_model=InputLogin), 
        output=JSON(),
        route='login'
)
def login(input: InputLogin) -> dict:
    """Responds with a JWT token if the username and password are valid."""


    if not (0 < input.timedelta_before_exp_sec <= 3600):
        raise InvalidArgument(
            "timedelta_before_exp_sec must be between 1 and 3600 seconds"
        )

    if input.username in USERS and USERS[input.username] == input.password:
        # Generate a JWT token for the user
        token = create_jwt_token(
            input.username, 
            input.timedelta_before_exp_sec
        )
        return {"token": token}
    else:
        raise UnauthorizedException("Invalid credentials")

# ---- ENDPOINT : PREDICTION ----

# input model
class InputModel(BaseModel):
    gre_score: int
    toefl_score: int
    university_rating: int
    sop: int
    lor: int
    cgpa: int
    research: int

# endpoint
@regression_model_service.api(
    input=JSON(pydantic_model=InputModel),
    output=JSON(),
    route='predict'
)
async def predict(input_data: InputModel) -> dict:
    
    # Convert the input data to a numpy array
    input_series = np.array([
        input_data.gre_score,
        input_data.toefl_score,
        input_data.university_rating,
        input_data.sop,
        input_data.lor,
        input_data.cgpa,
        input_data.research
    ])

    # predict using the regression model runner
    result = await regression_model_runner.predict.async_run(input_series.reshape(1, -1))

    return {"prediction": result.tolist()[0]}