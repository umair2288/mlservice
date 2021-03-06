FROM tiangolo/uwsgi-nginx-flask:python3.6
#RUN apt-get --update add nano
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY ./requirements.txt /var/www/requirements.txt
COPY mysettings.conf /etc/nginx/conf.d/
RUN pip install -r /var/www/requirements.txt