FROM python:3.6
WORKDIR /Users/dquarshie89/gitclone
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/dquarshie89/Data602-Assignment3
EXPOSE 5000
CMD [ "python", "/Users/dquarshie/gitclone/Data602-Assignment3/main.py" ]
