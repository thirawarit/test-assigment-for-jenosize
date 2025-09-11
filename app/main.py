from fastapi import FastAPI

from api.routes_users import router as routers_users
from core.middleware import log_requests

app = FastAPI()

# register middleware
app.middleware("http")(log_requests)

app.include_router(
    routers_users,
)

@app.get('/')
async def root():
    return {"message": "Hello World"}