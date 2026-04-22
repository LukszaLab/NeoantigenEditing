FROM ubuntu:24.04

LABEL maintainer="Nikhil Kumar (kumarn1@mskcc.org)" \
      version.neoantigenEditing="v1.3" \
      version.ubuntu="24.04" \
      version.python="3.11" \
      version.blast="2.12" \
      verion.biopython="1.81" \
      source.biopython="https://launchpad.net/ubuntu/+source/python-biopython/1.81+dfsg-1" \
      source.blast="https://launchpad.net/ubuntu/+source/ncbi-blast+/2.12.0+ds-3build1" \
      source.neoantigenEditing="https://github.com/mskcc/NeoantigenEditing" \
      source.credit_doi="https://doi.org/10.1038/s41586-022-04735-9"

ENV NEOANTIGEN_EDITING_TAG 1.3

RUN apt-get update \
  && apt-get install -y python3 python3-pip git make build-essential ncbi-blast+ python3-pandas python3-matplotlib python3-scipy python3-numpy python3-biopython \
  && cd /tmp \
  && git clone --branch ${NEOANTIGEN_EDITING_TAG} https://github.com/mskcc/NeoantigenEditing \
  && cd NeoantigenEditing \
  && cp -r . /usr/bin \
  && rm -r /tmp/NeoantigenEditing
