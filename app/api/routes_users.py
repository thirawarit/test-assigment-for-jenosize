from fastapi import APIRouter

from models.user_schema import InputItems
from services.pipeline import get_pipeline, get_generate
from services.prompt_builder import build_input_prompt

router = APIRouter(
    prefix="/users",          # ทุก endpoint ในไฟล์นี้จะ prefix ด้วย /users
    tags=["users"]            # ใช้จัดกลุ่มใน docs (Swagger UI)
)

@router.post("/service/")
async def response_content_from(input: InputItems):
    # pipe = get_pipeline(model_id=input.model_id)
    # prompt = build_input_prompt(input.model_dump())
    # output = pipe(prompt, max_new_tokens=100)[0]['generated_text']

    prompt = build_input_prompt(input.model_dump())
    output = get_generate(prompt=prompt, model_id=input.model_id, enable_thinking=True)
    response_body = input.model_dump()
    # response_body.update({'content': output})
    response_body.update(output)
    return response_body