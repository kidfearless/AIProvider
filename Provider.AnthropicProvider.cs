using AIProvider.Messages;

using Anthropic;

using Microsoft.Extensions.AI;

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

using System.Diagnostics.CodeAnalysis;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;

namespace AIProvider;

public abstract partial record Provider
{
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

            if (ApiKey is null)
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

                var apiKey = ApiKey!;
                using var chatClient = new AnthropicClient(apiKey)
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


            var response = await ChatClientStructuredOutputExtensions.GetResponseAsync<T>(chatClient, messages, chatOptions, useJsonSchemaResponseFormat: false);


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
#pragma warning disable OPENAI001 // Type is for evaluation purposes only and is subject to change or removal in future updates. Suppress this diagnostic to proceed.

            var messages = session.Messages.Where(m => m is not SystemPromptMessage).Select(m => m switch
            {
                UserMessage => new InputMessage(m.Content, InputMessageRole.User),
                AssistantMessage => new InputMessage(m.Content, InputMessageRole.Assistant),
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
