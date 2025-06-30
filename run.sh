nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu \
  --pytorch=autograd-nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -f true -x true \
  -o report-nsys \
  python bench.py --config config.ini