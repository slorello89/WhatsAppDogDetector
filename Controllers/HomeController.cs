using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json.Linq;
using Microsoft.Extensions.Configuration;
using WhatsAppDogDetector.Models;
using Vonage.Request;

namespace WhatsAppDogDetector.Controllers
{
    public class HomeController : Controller
    {
        private readonly IConfiguration _config;
        public static Dictionary<string, string> _pendingTrainLabels = new Dictionary<string, string>();

        public HomeController(IConfiguration config)
        {
            _config = config;
        }

        public IActionResult Index()
        {
            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [HttpPost("/webhooks/inbound")]
        public async Task<IActionResult> InboundMessage()
        {
            var message = await Vonage.Utility.WebhookParser.ParseWebhookAsync<JObject>(Request.Body, Request.ContentType);
            
            string responseMessage = string.Empty;
            if (message["message"]["content"]["type"].ToString() == "image")
            {
                var url = message["message"]["content"]["image"]["url"].ToString();
                if (_pendingTrainLabels.Keys.Contains(message["from"]["number"].ToString()))
                {
                    Detector.Instance.AddTrainingImage(url, _pendingTrainLabels[message["from"]["number"].ToString()]);
                    _pendingTrainLabels.Remove(message["from"]["number"].ToString());
                    responseMessage = "Training data added!";
                }
                else
                {
                    responseMessage = Detector.Instance.ClassifySingleImage(message["message"]["content"]["image"]["url"].ToString());
                }
                var to = new { type = "whatsapp", number = message["from"]["number"] };
                var from = new { type = "whatsapp", number = message["to"]["number"] };
                var content = new { type = "text", text = $"{ responseMessage }" };
                var msg = new { content };
                var request = new { to, from, message = msg};
                var creds = Credentials.FromAppIdAndPrivateKeyPath(_config["APP_ID"], _config["PRIVATE_KEY_PATH"]);
                var requestUri = new Uri("https://api.nexmo.com/v0.1/messages");
                var json = Newtonsoft.Json.JsonConvert.SerializeObject(request);
                Console.Write(json);
                await ApiRequest.DoRequestWithJsonContentAsync<JObject>
                (
                    method: "POST",
                    uri: requestUri,
                    payload: request,
                    authType: ApiRequest.AuthType.Bearer,
                    creds: creds
                );
            }
            else if (message["message"]["content"]["type"].ToString() == "text" && message["message"]["content"]["text"].ToString().Split(" ")[0].ToLower() == "train")
            {
                var dogType = message["message"]["content"]["text"].ToString().Split(" ")[1].ToLower();
                _pendingTrainLabels.Add(message["from"]["number"].ToString(), dogType);

            }
            return NoContent();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
