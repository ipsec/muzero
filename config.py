from pydantic import BaseSettings, BaseModel


class ResponseSettings(BaseModel):
    app_name: str = 'MuZero Tests'
    admin_email: str = 'fernando.ribeiro@gmail.com'

    class Config:
        orm_mode = True


class Settings(BaseSettings):
    app_name: str = 'MuZero Tests'
    admin_email: str = 'fernando.ribeiro@gmail.com'
    features: int = 11
    actions: int = 3
    memory: int = 100000
    batch: int = 4096


settings = Settings()
