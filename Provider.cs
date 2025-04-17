using AIProvider.Messages;

using Anthropic;

using Microsoft.Extensions.AI;

using OpenAI;

using System.Collections.ObjectModel;
using System.Linq;
using System.Net;
using System.Runtime.CompilerServices;
using System.Text;

namespace AIProvider;

public class DevProxy : IWebProxy
{
    public ICredentials? Credentials { get; set; }

    public Uri? GetProxy(Uri destination)
    {
        throw new NotImplementedException();
    }

    public bool IsBypassed(Uri host)
    {
        throw new NotImplementedException();
    }
}


public abstract record Provider : IDisposable
{
    protected abstract string Key { get; }
    protected abstract string Url { get; }

    protected string? ApiKey { get; set; }
    protected bool IsInitialized { get; set; }
    public List<AITool> Tools { get; private set; } = [];


    public virtual ChatSession CreateChatSession(ChatModel chatModel) => new(this, chatModel, StreamResponseAsync);
    public abstract Task<List<ChatModel>> GetModelsAsync();

    protected abstract IAsyncEnumerable<Response> StreamResponseAsync(ChatSession session, CancellationToken cancellationToken);

    public virtual void Initialize(string apiKey)
    {
        ApiKey = apiKey;
        if (ApiKey is null or "")
        {
            throw new Exception("API key not set");
        }

        IsInitialized = true;
    }

    public virtual void LoadTools(List<AITool> tools) => Tools = tools;

    public static Provider GetProvider(string key, string apiKey)
    {
        Provider provider = key switch
        {
            "OpenAI" => new OpenAiProvider(),
            "Anthropic" => new AnthropicProvider(),
            "Gemini" => new GeminiProvider(),
            _ => throw new Exception("Invalid provider")
        };

        provider.Initialize(apiKey);
        return provider;
    }

    public void Dispose()
    {
        throw new NotImplementedException();
    }

    public record OpenAiProvider : Provider
    {
        protected override string Key => "OpenAI";
        protected override string Url => "https://api.openai.com/v1/chat/";
        protected virtual OpenAI.OpenAIClientOptions Options => new() { Endpoint = new(Url) };

        public override void Initialize(string apiKey)
        {
            base.Initialize(apiKey);
        }
        public override async Task<List<ChatModel>> GetModelsAsync()
        {
            if (!IsInitialized)
            {
                throw new Exception("Provider not initialized");
            }

            var client = new OpenAIClient(ApiKey)
              .GetOpenAIModelClient();
            var models = await client.GetModelsAsync();
            return models.Value.Select(m => new ChatModel(m.Id)).ToList();
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
               new OpenAIClient(ApiKey)
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
                Tools = Tools
            };

            var response = await chatClient.GetResponseAsync(messages, chatOptions, cancellationToken: cancellationToken);

            var builder = new StringBuilder(2048);
            yield return new Response(response.Text);
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

    // uses the openai compatibility layer of gemini
    public record GeminiProvider : OpenAiProvider
    {
        protected override string Key => "Gemini";
        protected override string Url => "https://generativelanguage.googleapis.com/v1beta/openai/";
    }

    public record class AnthropicProvider : Provider
    {
        protected override string Key => "Anthropic";
        protected override string Url => "https://api.anthropic.com/v1/messages";

        public override async Task<List<ChatModel>> GetModelsAsync()
        {
            if (!IsInitialized || ApiKey is null)
            {
                throw new Exception("Provider not initialized");
            }

            AnthropicClient client = new(ApiKey);
            var models = await client.ModelsListAsync();
            return models.Data.Select(m => new ChatModel(m.Id)).ToList();
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
            var request = new CreateMessageParams()
            {
                Model = session.ChatModel.Model,
                MaxTokens = session.ChatModel.Model.Contains("haiku") ? 4096 : 8192,
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
    }

}
