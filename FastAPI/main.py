import uvicorn
from fastapi import FastAPI

# creating fastapi object

app = FastAPI()

# index route @http://127.0.0.1:8000
@app.get("/")
def index():
    return {'message' : 'Hello World!'}

# route with single parameter @http://127.0.0.1/param
@app.get('/welcome')
def get_name(name: str):
    return {'Welcome! ':f'{name}'}


# run the app
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# run:
# uvicorn main:app --reload
