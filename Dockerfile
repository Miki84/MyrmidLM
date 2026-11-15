# MyrmidLM master as a container: the router without a GPU (VM, NAS,
# Proxmox), models on the hosts. llama.cpp is built for the CPU only, for the
# embedding model of the router ("Beispiele") or a small model of its own.
#
#   docker run -d --name myrmid --restart unless-stopped --network host \
#     -v myrmid-data:/var/lib/myrmid -v myrmid-cache:/var/cache/myrmid \
#     ghcr.io/miki84/myrmidlm:latest
#
# --network host: the master finds hosts and workers by mDNS and wakes them by
# Wake-on-LAN, both need the LAN itself. Dashboard on port 8090, API on 8080.
FROM ubuntu:24.04 AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY VERSION ./
COPY node ./node
RUN bash node/build.sh --backends cpu --install /opt/myrmid/release

FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-zeroconf ca-certificates libgomp1 iproute2 curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/myrmid/release /opt/myrmid/current
ENV MYRMID_CONTAINER=1 \
    XDG_CONFIG_HOME=/var/lib/myrmid \
    XDG_CACHE_HOME=/var/cache/myrmid \
    PATH=/opt/myrmid/current/bin:$PATH
VOLUME ["/var/lib/myrmid", "/var/cache/myrmid"]
EXPOSE 8080 8090
ENTRYPOINT ["myrmid"]
CMD ["serve", "-m", "none", "--workers", "off"]
