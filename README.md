# deepstream-video-pipeline

## Dimension
- stage 1~5
- run / profile / debug?
- gpu 1/2
- host / device

## export model (TorchScript, TensorRT, ONNX)
    
    python export_tsc.py --ssd-module-name=ds_ssd300_1 --trt-module-name=ds_trt_1 --tsc-module-name=ds_tsc_1 --batch-dim=16 
    -> ds_tsc_1.tsc.pth.0, ds_tsc_1.tsc.pth.1 생성
    python export_trt_engine.py --ssd-module-name=ds_ssd300_1 --trt-module-name=ds_trt_1 --batch-dim=16 --output-names=image_nchw
    -> ds_trt_1.engine, ds_trt_1.onnx 생성

## Pipeline

    DS_TSC_INPUTS="$(shell python ds_trt_1.py)" 
    DS_TSC_PTH_PATH="checkpoints/ds_tsc_1.tsc.pth." 
    python ds_pipeline.py --batch-size=16 --name=1 --buffers=256 --gpus=1

## detail

    make run_pipeline_1_1gpu_host
    make logs/1_batch16.pipeline.dot

    python export_tsc.py --ssd-module-name=ds_ssd300_1 --trt-module-name=ds_trt_1 --tsc-module-name=ds_tsc_1 --batch-dim=16
    python export_trt_engine.py --ssd-module-name=ds_ssd300_1 --trt-module-name=ds_trt_1 --batch-dim=16 --output-names=image_nchw
    python ds_trt_1.py를 실행결과는 export_trt_engine.py에 output_names으로


    --- 
    checkpoints/ds_tsc_%.tsc.pth.0 checkpoints/ds_tsc_%.tsc.pth.1: export_tsc.py ds_tsc_%.py ds_trt_%.py ds_ssd300_%.py
    	python $< --ssd-module-name=ds_ssd300_$* --trt-module-name=ds_trt_$* --tsc-module-name=ds_tsc_$* --batch-dim=16
