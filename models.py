from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# =========================
# User Pydantic Models
# =========================

class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: EmailStr
    full_name: str
    role: str = Field(default="user", description="User role: admin, doctor, patient")


class UserCreate(UserBase):
    """Schema for creating a new user"""
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")


class UserUpdate(BaseModel):
    """Schema for updating user information"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = None


class UserResponse(UserBase):
    """Schema for returning user data (without password)"""
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
