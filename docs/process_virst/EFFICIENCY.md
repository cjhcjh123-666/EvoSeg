# ProcessVIRST efficiency record

The prototype trains 3,217,666 parameters: 3,217,154 in the Process
Conditioner and 512 in its fusion LayerNorm. VIRST, its large video/language
encoders, and SAM2 remain frozen.

Full-seed pilot training took 42.1--43.1 minutes for 384 single-sample updates
and peaked at 19.392--19.393 GiB allocated memory per worker.

GroundMoRe pilot inference, including decoding, VLM, ProcessVIRST, prompt
generation, and propagation, averaged 8.48 seconds per expression over the
three Full seeds. The official pilot run took 7.40 seconds per expression, so
the observed total overhead was about 1.08 seconds/expression (14.6%). Runs were
concurrent and this is not a formal isolated latency result.

Observed device memory during Full pilot inference was 30.7--34.1 GiB. This is
an `nvidia-smi` observation, not a synchronized allocator peak. The official
full baseline used about 33.0 GiB under its separate run. No efficiency claim is
made from these non-isolated measurements.
