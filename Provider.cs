using AIProvider.Messages;

using Anthropic;

using Microsoft.Extensions.AI;

using OpenAI;

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

using System.Diagnostics.CodeAnalysis;
using Microsoft.Extensions.Configuration;
using System.ClientModel;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using System;
using Microsoft.Extensions.Options;
using static System.Xml.Schema.XmlSchemaInference;

namespace AIProvider;

public abstract partial record Provider : IDisposable
{
    protected abstract string Key { get; }
    protected abstract string Url { get; }

    protected string? ApiKey { get; set; }
    protected bool IsInitialized { get; set; }
    public List<AITool> Tools { get; private set; } = [];
    protected IConfiguration? Configuration { get; set; }


    public virtual ChatSession CreateChatSession(ChatModel chatModel) => new ChatSession(this, chatModel);
    public abstract Task<List<ChatModel>> GetModelsAsync();

    protected abstract IAsyncEnumerable<Response> StreamResponseAsync(ChatSession session, CancellationToken cancellationToken);
    protected abstract Task<T> StructuredOutputAsync<T>(ChatSession session);

    public virtual void Initialize(string apiKey, IConfiguration? options = null)
    {
        ApiKey = apiKey;
        Configuration = options;

        IsInitialized = true;
    }

    public static Provider GetProvider(string key, string apiKey, IConfiguration? options = null)
    {
        Provider provider = key switch
        {
            "OpenAI" => new OpenAiProvider(),
            "Anthropic" => new AnthropicProvider(),
            "Gemini" => new GeminiProvider(),
            "AzureOpenAI" => new AzureProvider(),
            _ => throw new Exception("Invalid provider")
        };

        provider.Initialize(apiKey, options);
        return provider;
    }

    public virtual void LoadTools(List<AITool> tools) => Tools = tools;

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    public record OpenAiProvider : Provider
    {
        private string _url = "https://api.openai.com/v1/";

        protected override string Key => "OpenAI";
        protected override string Url => _url;
        protected virtual OpenAI.OpenAIClientOptions Options => new() { Endpoint = new(Url) };
        protected virtual ApiKeyCredential Auth => new(ApiKey!);

        public override void Initialize(string apiKey, IConfiguration? options = null)
        {
            base.Initialize(apiKey, options);
            if (options != null)
            {
                var url = options[$"Provider:{Key}:Url"];
                if (url is not null or "")
                {
                    _url = url;
                }
            }
        }


        public override async Task<List<ChatModel>> GetModelsAsync()
        {
            if (!IsInitialized)
            {
                throw new Exception("Provider not initialized");
            }

            var client = new OpenAIClient(Auth, Options)
              .GetOpenAIModelClient();
            var models = await client.GetModelsAsync();
            return models.Value.Select(m => new ChatModel(m.Id)).ToList();
        }

        protected override async Task<T> StructuredOutputAsync<T>(ChatSession session)
        {
            if (!IsInitialized)
            {
                throw new Exception("Provider not initialized");
            }

            if (!session.Messages.Any())
            {
                throw new Exception("No messages to send");
            }

            using var chatClient =
               new OpenAIClient(Auth, Options)
               .GetChatClient(session.ChatModel.Model)
               .AsIChatClient()
               .AsBuilder()
               .UseFunctionInvocation()
               .Build();


            var messages = session.Messages.Select(m =>
            {
                return m switch
                {
                    AssistantMessage a => ConvertToChatMessage(a),
                    SystemPromptMessage a => ConvertToChatMessage(a),
                    UserMessage a => ConvertToChatMessage(a),
                    Messages.Message a => ConvertToChatMessage(a),
                    _ => throw new NotImplementedException()
                };
            })
            .TakeLast(session.ShortTermMemoryLength + 1)
            .ToList();



            var chatOptions = new ChatOptions()
            {
                Tools = Tools,
                MaxOutputTokens = (int?)session.MaxOutputTokens
            };

            var response = await chatClient.GetResponseAsync<T>(messages, chatOptions);

            return response.TryGetResult(out var result) ? result : throw new Exception("Failed to get structured output");
        }


        protected override async IAsyncEnumerable<Response> StreamResponseAsync(ChatSession session, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            if (!IsInitialized)
            {
                throw new Exception("Provider not initialized");
            }

            if (!session.Messages.Any())
            {
                throw new Exception("No messages to send");
            }

            using var chatClient =
               new OpenAIClient(Auth, Options)
               .GetChatClient(session.ChatModel.Model)
               .AsIChatClient()
               .AsBuilder()
               .UseFunctionInvocation()
               .Build();


            var messages = session.Messages.Select(m =>
            {
                return m switch
                {
                    AssistantMessage a => ConvertToChatMessage(a),
                    SystemPromptMessage a => ConvertToChatMessage(a),
                    UserMessage a => ConvertToChatMessage(a),
                    Messages.Message a => ConvertToChatMessage(a),
                    _ => throw new NotImplementedException()
                };
            })
            .TakeLast(session.ShortTermMemoryLength + 1)
            .ToList();



            var chatOptions = new ChatOptions()
            {
                Tools = Tools,
                MaxOutputTokens = (int?)session.MaxOutputTokens
            };

            var stream = chatClient.GetStreamingResponseAsync(messages, chatOptions, cancellationToken: cancellationToken);
            await foreach (var message in stream)
            {
                foreach (var contentPart in message.Contents.OfType<TextContent>())
                {
                    if (contentPart.Text != null)
                    {
                        yield return new(contentPart.Text);
                    }
                }
            }
        }

        protected ChatMessage ConvertToChatMessage(Messages.Message message) => new Microsoft.Extensions.AI.ChatMessage(new(message.Role), message.Content);
        protected Microsoft.Extensions.AI.ChatMessage ConvertToChatMessage(Messages.AssistantMessage message) => new Microsoft.Extensions.AI.ChatMessage(ChatRole.Assistant, message.Content);
        protected Microsoft.Extensions.AI.ChatMessage ConvertToChatMessage(Messages.SystemPromptMessage message) => new Microsoft.Extensions.AI.ChatMessage(ChatRole.System, message.Content);
        protected Microsoft.Extensions.AI.ChatMessage ConvertToChatMessage(Messages.UserMessage message)
        {
            var chatMessage = new Microsoft.Extensions.AI.ChatMessage(ChatRole.User, message.Content);
            message.Files.ForEach(chatMessage.Contents.Add);

            return chatMessage;
        }

    }

    // uses the openai compatibility layer of gemini
    public record GeminiProvider : OpenAiProvider
    {
        protected override string Key => "Gemini";
        protected override string Url => "https://generativelanguage.googleapis.com/v1beta/openai/";
    }

    public record class AnthropicProvider : Provider
    {
        protected override string Key => "Anthropic";
        protected override string Url => "https://api.anthropic.com/v1/";

        public override async Task<List<ChatModel>> GetModelsAsync()
        {
            if (!IsInitialized)
            {
                throw new Exception("Provider not initialized");
            }

            AnthropicClient client = new(ApiKey);
            var models = await client.ModelsListAsync();
            return models.Data.Select(m => new ChatModel(m.Id)).ToList();
        }

        [RequiresUnreferencedCode("serializer go brr")]
        [RequiresDynamicCode("serializer go brr")]
        protected override async Task<T> StructuredOutputAsync<T>(ChatSession session)
        {
            if (!IsInitialized)
            {
                throw new Exception("Provider not initialized");
            }

            if (!session.Messages.Any())
            {
                throw new Exception("No messages to send");
            }

            using var chatClient = new AnthropicClient(ApiKey!)
               .AsBuilder()
               .UseFunctionInvocation()
               .Build();


            var messages = session.Messages.Select(m =>
            {
                return m switch
                {
                    AssistantMessage a => ConvertToChatMessage(a),
                    SystemPromptMessage a => ConvertToChatMessage(a),
                    UserMessage a => ConvertToChatMessage(a),
                    Messages.Message a => ConvertToChatMessage(a),
                    _ => throw new NotImplementedException()
                };
            })
            .TakeLast(session.ShortTermMemoryLength + 1)
            .ToList();


            var chatOptions = new ChatOptions()
            {
                Tools = Tools,
                ModelId = session.ChatModel.Model
            };


            var response = await ChatClientStructuredOutputExtensions.GetResponseAsync<T>(chatClient, messages, chatOptions, useJsonSchema: false);


            if (response.TryGetResult(out var result))
            {
                return result;
            }

            var jsonText = response.Text.GetCodeBlockOrText();
            var jsonElement = System.Text.Json.JsonSerializer.Deserialize<T>(jsonText, JsonSerializerOptions.Web);
            if (jsonElement is not null)
            {
                return jsonElement;
            }

            throw new Exception("Failed to deserialize response");

        }

        protected override async IAsyncEnumerable<Response> StreamResponseAsync(ChatSession session, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            if (!IsInitialized || ApiKey is null)
            {
                throw new Exception("Provider not initialized");
            }


            if (!session.Messages.Any())
            {
                throw new Exception("No messages to send");
            }

            AnthropicClient client = new(ApiKey);

            var messages = session.Messages.Where(m => m is not SystemPromptMessage).Select(m => m switch
            {
                UserMessage => new InputMessage(InputMessageRole.User, m.Content),
                AssistantMessage => new InputMessage(InputMessageRole.Assistant, m.Content),
                _ => throw new Exception($"Invalid message type {m.GetType().Name} {m.Content}")
            }).TakeLast(session.ShortTermMemoryLength).ToList();

            var systemMessage = session.Messages.OfType<SystemPromptMessage>().FirstOrDefault()?.Content ?? "";
            var thinking = session.ChatModel.Model.Contains("claude-3-7-sonnet");

            int maxTokens = 8192;
            if (session.MaxOutputTokens.HasValue)
            {
                maxTokens = Math.Min(8192, (int)session.MaxOutputTokens.Value);
            }

            if (session.ChatModel.Model.Contains("haiku"))
            {
                maxTokens = Math.Min(4096, maxTokens);
            }
            var request = new CreateMessageParams()
            {
                Model = session.ChatModel.Model,
                MaxTokens = maxTokens,
                System = systemMessage,
                Thinking = thinking ? new() { Enabled = new(1025, ThinkingConfigEnabledType.Enabled) } : null,
                Temperature = 1,
                Messages = messages
            };

            var res = client.CreateMessageAsStreamAsync(request, cancellationToken: cancellationToken);

            var builder = new StringBuilder(2048);
            await foreach (var r in res)
            {
                var text = r.ContentBlockDelta?.Delta.TextDelta?.Text;

                if (text is not null)
                {
                    builder.Append(text);
                    yield return new Response(text);
                }
            }
        }


        protected ChatMessage ConvertToChatMessage(Messages.Message message) => new ChatMessage(new(message.Role), message.Content);
        protected ChatMessage ConvertToChatMessage(AssistantMessage message) => new ChatMessage(ChatRole.Assistant, message.Content);
        protected ChatMessage ConvertToChatMessage(SystemPromptMessage message) => new ChatMessage(ChatRole.System, message.Content);
        protected ChatMessage ConvertToChatMessage(UserMessage message)
        {
            var chatMessage = new ChatMessage(ChatRole.User, message.Content);
            message.Files.ForEach(chatMessage.Contents.Add);

            return chatMessage;
        }
    }

}
