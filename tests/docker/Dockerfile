FROM mcr.microsoft.com/mssql/server:latest
LABEL Name=mssql Version=0.0.1

ENV SA_PASSWORD Ahojdatadata123
ENV ACCEPT_EULA Y
ENV MSSQL_PID Express

EXPOSE 1433

USER root

CMD /bin/bash ./entrypoint.sh

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app
RUN chmod +x /usr/src/app/run-initialization.sh
