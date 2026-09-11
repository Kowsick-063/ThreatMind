from sqlalchemy import text

from backend.database import engine

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        print("Database connected successfully!")
        print(result.scalar())

except Exception as error:
    print("Database connection failed!")
    print(error)