from typing import List, Union
from pydantic import BaseModel, Field

class InputItems(BaseModel):
    model_id: str = Field(default='Thirawarit/Jenosize-Article-Qwen3-8b', description="a repo Hugging Face model")
    topic_category: List[str] = Field(default=None, description="หมวดหัวข้อที่เกี่ยวข้อง")
    industry: List[str] = Field(default=None, description="กลุ่มอุตสาหกรรม")
    target_audience: List[str] = Field(default=None, description="กลุ่มเป้าหมาย")
    website: str = Field(default=None, description="เว็บไซต์หรือ URL")
    seo_keywords: List[str] = Field(default=None, description="คำค้น SEO ที่เกี่ยวข้อง")