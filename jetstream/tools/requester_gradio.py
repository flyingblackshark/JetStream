# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A test request."""

import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
import jax.numpy as jnp
import jax
import dac_jax
from functools import partial
import gradio as gr
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jetstream.engine.token_utils import take_nearest_length
cc.set_cache_dir("./jax_cache")
DEFAULT_DECODE_BUCKETS = [
    64 * 1,
    64 * 2,
    64 * 4,
    64 * 8,
    64 * 16,
    64 * 32,
    64 * 64,
]
from transformers import AutoTokenizer

if __name__ == "__main__":
  jax.distributed.initialize()
  device_mesh = mesh_utils.create_device_mesh((1,1,4))
  mesh = Mesh(device_mesh, axis_names=("data", "model","tensor")) 
  model, variables = dac_jax.load_model(model_type="44khz")
  def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
    """
    Get a NamedSharding for a given PartitionSpec, and the device mesh.
    A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
    """
    return NamedSharding(mesh, pspec)
  x_sharding = get_sharding_for_spec(PartitionSpec("data", "model","tensor"))

  def _GetResponseAsync(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    request: jetstream_pb2.DecodeRequest,
  ):
    """Gets an async response."""

    response = stub.Decode(request)
    output = []
    codes_arr = []
    for resp in response:
      test = resp.stream_content.samples[0].token_ids
      output.append(resp.stream_content.samples[0].token_ids)
      if len(list(test)) > 0:
        codes_arr.append(list(test)[0].semantic_ids)
    codes = jnp.asarray(codes_arr)
    true_length = codes.shape[0]
    padded_length = take_nearest_length(DEFAULT_DECODE_BUCKETS, true_length)
    padding = padded_length - true_length
    codes = jnp.pad(codes,((0,padding),(0,0)))

    @partial(jax.jit, static_argnums=(1, 2), in_shardings=x_sharding,out_shardings=x_sharding)
    def decode_from_codes(codes: jnp.ndarray, scale, length: int = None):
        recons = model.apply(
            variables,
            codes,
            scale,
            length,
            method="decode",
        )
        return recons
    audio_output = decode_from_codes(jnp.expand_dims(codes.transpose(1,0),0),None).squeeze((0,1) )
    audio_output = audio_output[:true_length*512]
    return 44100,audio_output
  def main(text):
    address = "10.130.0.53:9000"
    with grpc.insecure_channel(address) as channel:
      grpc.channel_ready_future(channel).result()
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)
      print(f"Sending request to: {address}")
      tokenizer = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
      token_ids = tokenizer.encode(text)
      request = jetstream_pb2.DecodeRequest(
          token_content=jetstream_pb2.DecodeRequest.TokenContent(
              token_ids=token_ids
          ),
          max_tokens=4096,
      )
      return _GetResponseAsync(stub, request)
  iface = gr.Interface(
      fn=main,  # 处理函数
      inputs="text",      # 输入类型
      outputs="audio",    # 输出类型
      title="文本到语音转换器",  # 界面标题
      description="输入文本，将其转换为语音",  # 界面描述
      )

      # 启动 Gradio 界面
  iface.launch()
