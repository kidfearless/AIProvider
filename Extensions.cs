using AIProvider.Messages;

using ModelContextProtocol.Protocol.Types;

using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

namespace AIProvider;

public static class Extensions
{
    public static string GetCodeBlockOrText(this Response content) => content.Content.GetCodeBlockOrText();
    public static string GetCodeBlockOrText(this string content)
    {
        var regex = new Regex(@"```(?:\w*\n|\w*)(.*?)```", RegexOptions.Singleline);
        var match = regex.Match(content);
        if (match.Success)
        {
            return match.Groups[1].Value.Trim();
        }
        return content!.Trim();
    }

    public static List<Message> AddSystemPrompt(this List<Message> messages, string prompt)
    {
        messages.Add(new Messages.SystemPromptMessage(prompt));
        return messages;
    }

    public static List<Message> AddUserMessage(this List<Message> messages, string content)
    {
        messages.Add(new Messages.UserMessage(content));
        return messages;
    }

    public static List<Message> AddAssistantMessage(this List<Message> messages, string content)
    {
        messages.Add(new Messages.AssistantMessage(content));
        return messages;
    }

}
