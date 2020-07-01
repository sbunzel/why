ARG PYTHON_IMAGE_TAG=3.8-slim-buster

#
# First stage: get dependencies, build the project, store binaries (whl)
#

FROM python:${PYTHON_IMAGE_TAG} as py-build
# There are no pre-built wheels for shap, thus we need gcc
RUN apt-get update && apt-get install -y build-essential
WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt && pip wheel -r requirements.txt -w deps
COPY . .
RUN python setup.py install && python setup.py bdist_wheel

#
# Second stage: create the smallest possible image for deployment
#

FROM python:${PYTHON_IMAGE_TAG}
LABEL maintainer="Steffen Bunzel"
WORKDIR /app
COPY --from=py-build /build/deps/ deps/
RUN [ -n "$(ls -A deps)" ] && pip install deps/*.whl && rm -rf deps || echo "no dependencies to install"
COPY --from=py-build /build/dist/ dist/
RUN pip install dist/*.whl && rm -rf dist
COPY src/app.py ./
ENTRYPOINT ["streamlit", "run", "app.py"]