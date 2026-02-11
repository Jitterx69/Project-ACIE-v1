"""
Security and authentication module for ACIE API
Implements JWT authentication, API key management, and role-based access control
"""

from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
API_KEY_HEADER_NAME = "X-API-Key"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_security = HTTPBearer()
api_key_security = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


# ============================================================================
# Models
# ============================================================================

class User(BaseModel):
    """User model"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = ["user"]


class UserInDB(User):
    """User model with hashed password"""
    hashed_password: str


class Token(BaseModel):
    """Token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    roles: List[str] = []


# ============================================================================
# Mock Database (Replace with real database in production)
# ============================================================================

# In production, replace this with database queries
FAKE_USERS_DB = {
    "admin": UserInDB(
        username="admin",
        email="admin@acie.ai",
        full_name="ACIE Admin",
        hashed_password=pwd_context.hash("admin123"),
        roles=["admin", "user"]
    ),
    "user": UserInDB(
        username="user",
        email="user@acie.ai",
        full_name="ACIE User",
        hashed_password=pwd_context.hash("user123"),
        roles=["user"]
    )
}

# API keys (in production, store in database with hashing)
VALID_API_KEYS = {
    "ak_test_1234567890": {"name": "Test API Key", "roles": ["user"]},
    "ak_admin_9876543210": {"name": "Admin API Key", "roles": ["admin", "user"]}
}


# ============================================================================
# Authentication Functions
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    if username in FAKE_USERS_DB:
        return FAKE_USERS_DB[username]
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        roles: List[str] = payload.get("roles", [])
        
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return TokenData(username=username, roles=roles)
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# ============================================================================
# Dependency Functions
# ============================================================================

async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_security)
) -> User:
    """Get current user from JWT token"""
    token = credentials.credentials
    token_data = decode_token(token)
    
    user = get_user(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return User(**user.dict())


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(api_key_security)
) -> Optional[User]:
    """Get current user from API key"""
    if api_key is None:
        return None
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )
    
    key_data = VALID_API_KEYS[api_key]
    return User(
        username=key_data["name"],
        roles=key_data["roles"]
    )


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """Get current user from either JWT token or API key"""
    user = token_user or api_key_user
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


def require_role(required_role: str):
    """Dependency to require specific role"""
    async def role_checker(current_user: User = Depends(get_current_user)):
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return current_user
    
    return role_checker


# ============================================================================
# Public Authentication Endpoints (to be added to FastAPI app)
# ============================================================================

async def login_endpoint(username: str, password: str) -> Token:
    """Login endpoint to get JWT token"""
    user = authenticate_user(username, password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "roles": user.roles},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
