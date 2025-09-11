from fastapi import APIRouter, Depends

from models.user_schema import UserResponse, InputItems

router = APIRouter(
    prefix="/users",          # ทุก endpoint ในไฟล์นี้จะ prefix ด้วย /users
    tags=["users"]            # ใช้จัดกลุ่มใน docs (Swagger UI)
)

@router.post("/service/")
async def response_content_from(input: InputItems):
    return input