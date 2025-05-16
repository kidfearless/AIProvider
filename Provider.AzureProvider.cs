using AIProvider.Messages;

using Microsoft.Extensions.AI;

using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using Azure.AI.OpenAI;
using Azure;
using ChatResponseFormat = OpenAI.Chat.ChatResponseFormat;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.ComponentModel;
using System.Reflection;
using OpenAI.Chat;
using Newtonsoft.Json.Schema;
using Newtonsoft.Json;

namespace AIProvider;

public abstract partial record Provider
{
    public record AzureProvider : Provider
    {
        private string _url = "https://eastus.api.cognitive.microsoft.com/";
        protected override string Key => "AzureOpenAI";
        protected override string Url => _url;

        public override void Initialize(string apiKey, IConfiguration? options = null)
        {
            base.Initialize(apiKey);

            if (options != null)
            {
                var url = options[$"Provider:{Key}:Url"];
                if (url is not null or "")
                {
                    _url = url;
                }
            }
        }

        public record AzureChatSession(AzureProvider AzureProvider, ChatModel ChatModel) : Provider.ChatSession(AzureProvider, ChatModel)
        {
            public override Task<Response> GetResponseAsync()
            {
                return AzureProvider.GetResponseAsync(this);
            }
        }

        public override ChatSession CreateChatSession(ChatModel chatModel)
        {
            return new AzureChatSession(this, chatModel);
        }

        public override Task<List<ChatModel>> GetModelsAsync()
        {
            return Task.FromResult<List<ChatModel>>([]);
        }

        protected virtual async Task<Response> GetResponseAsync(ChatSession session)
        {
            if (!IsInitialized)
                throw new Exception("Provider not initialized");

            if (!session.Messages.Any())
                throw new Exception("No messages to send");

            var model = session.ChatModel.Model;


            var chatClient = new AzureOpenAIClient(new(Url!), new AzureKeyCredential(ApiKey!))
                .GetChatClient(model);

            var messages = session.Messages
                .Select(m => m switch
                {
                    AssistantMessage a => ConvertToChatMessage(a),
                    SystemPromptMessage a => ConvertToChatMessage(a),
                    UserMessage a => ConvertToChatMessage(a),
                    _ => throw new NotImplementedException()
                })
                .TakeLast(session.ShortTermMemoryLength + 1)
                .ToList();



            var chatOptions = new OpenAI.Chat.ChatCompletionOptions()
            {
            };

            var stream = await chatClient.CompleteChatAsync(messages, chatOptions);
            var joined = string.Concat(stream.Value.Content.Select(t => t.Text));
            return new Response(joined);
        }

        protected override async Task<T> StructuredOutputAsync<T>(ChatSession session)
        {
            if (!IsInitialized)
                throw new Exception("Provider not initialized");

            if (!session.Messages.Any())
                throw new Exception("No messages to send");

            var model = session.ChatModel.Model;


            var chatClient = new AzureOpenAIClient(new(Url!), new AzureKeyCredential(ApiKey!))
                .GetChatClient(model);

            var messages = session.Messages
                .Select(m => m switch
                {
                    AssistantMessage a => ConvertToChatMessage(a),
                    SystemPromptMessage a => ConvertToChatMessage(a),
                    UserMessage a => ConvertToChatMessage(a),
                    _ => throw new NotImplementedException()
                })
                .TakeLast(session.ShortTermMemoryLength + 1)
                .ToList();

            var options = new ChatCompletionOptions()
            {
                MaxOutputTokenCount = session.MaxOutputTokens.HasValue ? (int?)session.MaxOutputTokens.Value : null
            };

            var stream = await GetResponseAsync<T>(chatClient, messages, AIJsonUtilities.DefaultOptions, options);
            var text = stream.Value.Content[0].Text.GetCodeBlockOrText();

            return JsonConvert.DeserializeObject<T>(text) ??
                throw new InvalidOperationException("Could not deserialize response");
        }

        protected async Task<System.ClientModel.ClientResult<ChatCompletion>> GetResponseAsync<T>(ChatClient chatClient, IEnumerable<OpenAI.Chat.ChatMessage> messages, JsonSerializerOptions serializerOptions, ChatCompletionOptions? options = null, bool? useJsonSchema = null, CancellationToken cancellationToken = default(CancellationToken))
        {
            serializerOptions.MakeReadOnly();
            JsonElement jsonElement = AIJsonUtilities.CreateJsonSchema(typeof(T), null, hasDefaultValue: false, null, serializerOptions, new AIJsonSchemaCreateOptions
            {
                IncludeSchemaKeyword = true,
                DisallowAdditionalProperties = true,
                IncludeTypeInEnumSchemas = true
            });
            bool isWrappedInObject;
            JsonElement jsonElement2;
            if (SchemaRepresentsObject(jsonElement))
            {
                isWrappedInObject = false;
                jsonElement2 = jsonElement;
            }
            else
            {
                isWrappedInObject = true;
                JsonObject obj = new JsonObject
                {
                    { "$schema", "https://json-schema.org/draft/2020-12/schema" },
                    { "type", "object" },
                    {
                        "properties",
                        new JsonObject {
                        {
                            "data",
                            JsonElementToJsonNode(jsonElement)
                        } }
                    },
                    {
                        "additionalProperties",
                        (JsonNode)false
                    }
                };
                JsonNode reference = "data";
                obj.Add("required", new JsonArray(new ReadOnlySpan<JsonNode>(ref reference)));
                jsonElement2 = System.Text.Json.JsonSerializer.SerializeToElement(obj, AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(JsonObject)));
            }
            options = ((options != null) ? options : new ChatCompletionOptions());
            if (useJsonSchema.GetValueOrDefault(true))
            {
                options.ResponseFormat = ChatResponseFormat.CreateJsonSchemaFormat("response",BinaryData.FromString(jsonElement2.GetRawText()));
            }
            else
            {
                options.ResponseFormat = ChatResponseFormat.CreateJsonObjectFormat();
                var item = new OpenAI.Chat.UserChatMessage($"Respond with a JSON value conforming to the following schema:\r\n```\r\n{jsonElement2}\r\n```");
                var list = new List<OpenAI.Chat.ChatMessage>();
                list.AddRange(messages);
                list.Add(item);
                messages = list;
            }

            var response = await chatClient.CompleteChatAsync(messages, options);
            return response;
        }

        private static bool SchemaRepresentsObject(JsonElement schemaElement)
        {
            if (schemaElement.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty item in schemaElement.EnumerateObject())
                {
                    if (item.NameEquals("type"u8))
                    {   
                        return item.Value.ValueKind == JsonValueKind.String && item.Value.ValueEquals("object"u8);
                    }
                }
            }

            return false;
        }

        private static JsonNode? JsonElementToJsonNode(JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.Null => null,
                JsonValueKind.Array => JsonArray.Create(element),
                JsonValueKind.Object => JsonObject.Create(element),
                _ => JsonValue.Create(element),
            };
        }


        protected override async IAsyncEnumerable<Response> StreamResponseAsync(
            ChatSession session,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            if (!IsInitialized)
                throw new Exception("Provider not initialized");

            if (!session.Messages.Any())
                throw new Exception("No messages to send");


            var chatClient = new AzureOpenAIClient(new(Url!), new AzureKeyCredential(ApiKey!))
                .GetChatClient(session.ChatModel.Model);

            var messages = session.Messages
                .Select(m => m switch
                {
                    AssistantMessage a => ConvertToChatMessage(a),
                    SystemPromptMessage a => ConvertToChatMessage(a),
                    UserMessage a => ConvertToChatMessage(a),
                    _ => throw new NotImplementedException()
                })
                .TakeLast(session.ShortTermMemoryLength + 1)
                .ToList();

            var chatOptions = new ChatCompletionOptions()
            {
            };

            var stream = chatClient.CompleteChatStreamingAsync(messages, chatOptions, cancellationToken);

            await foreach (var message in stream)
            {
                foreach (var contentPart in message.ContentUpdate)
                {
                    if (contentPart.Text != null)
                    {
                        yield return new(contentPart.Text);
                    }
                }
            }
        }

        protected OpenAI.Chat.ChatMessage ConvertToChatMessage(AssistantMessage message) =>
            OpenAI.Chat.AssistantChatMessage.CreateAssistantMessage(message.Content);

        protected OpenAI.Chat.ChatMessage ConvertToChatMessage(SystemPromptMessage message) =>
            OpenAI.Chat.SystemChatMessage.CreateAssistantMessage(message.Content);

        protected OpenAI.Chat.ChatMessage ConvertToChatMessage(UserMessage message) =>
            new OpenAI.Chat.UserChatMessage(message.Content);
    }

}
