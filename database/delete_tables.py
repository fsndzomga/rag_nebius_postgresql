from db import engine, Base
from sqlalchemy.exc import SQLAlchemyError


def delete_all_tables():
    try:
        # Drop all tables defined in the Base metadata
        Base.metadata.drop_all(bind=engine)
        print("All tables have been deleted successfully.")
    except SQLAlchemyError as e:
        print(f"Error while deleting tables: {e}")


if __name__ == "__main__":
    delete_all_tables()
