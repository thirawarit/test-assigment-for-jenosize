import time
import datetime
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn.access")

# ----------------------------
# Logging middleware
# ----------------------------
async def log_requests(request: Request, call_next):
    start_time = time.time()
    date = datetime.datetime.now().strftime('%d-%h-%Y %H:%M:%S')
    response = await call_next(request)
    process_time = time.time() - start_time
    print(
        f"{request.method} {request.url.path} at {date} completed in {process_time:.2f}s "
        f"status_code={response.status_code}"
    )
    return response

# # ----------------------------
# # Auth middleware (JWT example)
# # ----------------------------
# from jose import jwt, JWTError

# SECRET_KEY = "supersecret"
# ALGORITHM = "HS256"

# async def auth_middleware(request: Request, call_next):
#     # Skip public endpoints
#     if request.url.path.startswith("/public"):
#         return await call_next(request)

#     token = request.headers.get("Authorization")
#     if not token or not token.startswith("Bearer "):
#         return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

#     try:
#         payload = jwt.decode(token.split(" ")[1], SECRET_KEY, algorithms=[ALGORITHM])
#         request.state.user = payload  # store user info in request.state
#     except JWTError:
#         return JSONResponse(status_code=401, content={"detail": "Invalid token"})

#     return await call_next(request)
