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

from typing import Sequence

from absl import app
from absl import flags
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
import jax.numpy as jnp
import jax
import dac_jax
import soundfile as sf
from functools import partial
import gradio as gr
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
MAX_LENGTH = 44100 // 512 * 60
# from jetstream.engine.token_utils import load_vocab

# _SERVER = flags.DEFINE_string("server", "35.186.132.71", "server address")
# _PORT = flags.DEFINE_string("port", "9000", "port to ping")
# #_TEXT = flags.DEFINE_string("text", "This is a test.", "The message")
# _MAX_TOKENS = flags.DEFINE_integer(
#     "max_tokens", 4096, "Maximum number of output/decode tokens of a sequence"
# )
# _TOKENIZER = flags.DEFINE_string(
#     "tokenizer",
#     None,
#     "Name or path of the tokenizer (matched to the model)",
#     required=True,
# )
# _CLIENT_SIDE_TOKENIZATION = flags.DEFINE_bool(
#     "client_side_tokenization",
#     False,
#     "Enable client side tokenization with tokenizer.",  
# )


def _GetResponseAsync(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    request: jetstream_pb2.DecodeRequest,
):
  """Gets an async response."""

  response = stub.Decode(request)
  output = []
  codes_arr = []
  for resp in response:
    # if _CLIENT_SIDE_TOKENIZATION.value:
    test = resp.stream_content.samples[0].token_ids
    #print(test)
    output.append(resp.stream_content.samples[0].token_ids)
    if len(list(test)) > 0:
      codes_arr.append(list(test)[0].semantic_ids)

    # else:
    #   output.extend(resp.stream_content.samples[0].text)
  # for i in range(len(output)):
  #   test = 
  #   codes_arr.append()
  codes = jnp.asarray(codes_arr)
  real_length = codes.shape[0]
  padding = MAX_LENGTH - real_length
  codes = jnp.pad(codes,((0,padding),(0,0)))
  model, variables = dac_jax.load_model(model_type="44khz")
  @partial(jax.jit, static_argnums=(1, 2))
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
  audio_output = audio_output[:real_length*512]
  return 44100,audio_output
  #sf.write("test.wav",audio_output,samplerate=44100)
  
  # if _CLIENT_SIDE_TOKENIZATION.value:
  # vocab = load_vocab(_TOKENIZER.value)
  # text_output = vocab.tokenizer.decode(output)
  # else:
  #   text_output = "".join(output)
  # print(f"Prompt: {_TEXT.value}")
  # print(f"Response: {output}")
  


from transformers import AutoTokenizer





if __name__ == "__main__":
  #app.run(main)
  def main(text):
    # Note: Uses insecure_channel only for local testing. Please add grpc
    # credentials for Production.
    address = "35.186.132.71:9000"
    with grpc.insecure_channel(address) as channel:
      grpc.channel_ready_future(channel).result()
      stub = jetstream_pb2_grpc.OrchestratorStub(channel)
      print(f"Sending request to: {address}")
      # if _CLIENT_SIDE_TOKENIZATION.value:
      # vocab = load_vocab(_TOKENIZER.value)
      # token_ids = vocab.tokenizer.encode(_TEXT.value)
      tokenizer = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
      token_ids = tokenizer.encode(text)
      request = jetstream_pb2.DecodeRequest(
          token_content=jetstream_pb2.DecodeRequest.TokenContent(
              token_ids=token_ids
          ),
          max_tokens=4096,
      )
      # else:
      #   request = jetstream_pb2.DecodeRequest(
      #       text_content=jetstream_pb2.DecodeRequest.TextContent(
      #           text=_TEXT.value
      #       ),
      #       max_tokens=_MAX_TOKENS.value,
      #   )
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
