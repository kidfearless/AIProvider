using AIProvider.Messages;

using Microsoft.Extensions.AI;

using OpenAI;

using System.Runtime.CompilerServices;
using Microsoft.Extensions.Configuration;
using System.ClientModel;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using System.Diagnostics.CodeAnalysis;

namespace AIProvider;

public abstract partial record Provider
{
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

}
