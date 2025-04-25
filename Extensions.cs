using ModelContextProtocol.Protocol.Types;

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

}
