docker run \
    -it \
    -v $(pwd):/workspaces/monoUWNet \
    -w /workspaces/monoUWNet \
    --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    viseaon/monou \
    /workspaces/monoUWNet/evaluate.sh
    # -e DISPLAY=${DISPLAY} \
    # -e QT_X11_NO_MITSHM=1 \
    # -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    # -e XAUTHORITY=${XAUTH} \
    # -v ${XAUTH}:${XAUTH} \