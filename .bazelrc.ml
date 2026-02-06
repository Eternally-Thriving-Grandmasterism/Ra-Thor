# .bazelrc.ml – ML build optimizations for TensorFlow / PyTorch

build:ml --config=ml
build:ml --copt=-march=native
build:ml --linkopt=-march=native
build:ml --define=USE_CUDA=1
build:ml --action_env=CUDA_VISIBLE_DEVICES=0,1
build:ml --jobs=auto
test:ml --test_output=errors
run:ml --test_output=errors

# Remote execution (optional – RBE)
build:rbe --remote_executor=grpcs://remotebuild.googleapis.com
build:rbe --remote_instance_name=projects/your-project/locations/us-west1/instances/default_instance
build:rbe --google_default_credentials
