FROM godatadriven/pyspark:3.4

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install quinn
RUN pip install numpy
RUN pip install pandas

COPY Wine_Application.py /app/Wine_Application.py

WORKDIR /app

ENTRYPOINT ["python", "Wine_Application.py"]
