from django.shortcuts import render


sump_url= "http://127.0.0.1:7575/chatbot"

sump_file_url= "http://127.0.0.1:7575/upload_file"


def sump_chat_bot(request):
    return render(request,"chatbot.html",{
            "chatbot":sump_url,
            "file_url":sump_file_url or "",
            "mode":"vac"
            }
            )


cnam_url= "http://127.0.0.1:6700/chatbot"

def cnam_chat_bot(request):
    return render(request,"chatbot.html",{
            "chatbot":cnam_url,
            "mode":"cnam"
            }
            )


vac_url= "http://127.0.0.1:6700/chatbot"


def vac_chat_bot(request):
    return render(request,"chatbot.html",{
            "chatbot":vac_url,
            "mode":"vac"
            }
            )


ord_url= "http://127.0.0.1:7989/llm"

def ord_chat_bot(request):
    return render(request,"chatbot.html",{
            "chatbot":ord_url,
            "mode":"ord"
            }
            )


pdf_url= "http://127.0.0.1:7989/llm"

def pdf_chat_bot(request):
    return render(request,"chatbot.html",{
            "chatbot":pdf_url,
            "mode":"cnam"
            }
            )

LUNG_UPLOAD_URL= "http://127.0.0.1:7777/lung_predict"
def lung_predict(request):
    return render(request,"lung_pred.html",{
            "UPLOAD_URL":LUNG_UPLOAD_URL,
            "PROCESS_URL" : sump_url
            }
            )


CARRIE_UPLOAD_URL= "http://127.0.0.1:7777/carrie_predict"
def carrie_predict(request):
    return render(request,"carrie_pred.html",{
            "UPLOAD_URL":CARRIE_UPLOAD_URL,
            "PROCESS_URL" : sump_url

            }
            )


def about(request):
    return render(request,"about-us.html")


def blog(request):
    return render(request,"blog.html")


def contact(request):
    return render(request,"contact.html")


def department(request):
    return render(request,"department.html")

    
def doctors(request):
    return render(request,"doctors.html")


    
def element(request):
    return render(request,"element.html")


    
def index(request):
    return render(request,"index.html")



    
def singleblog(request):
    return render(request,"single-blog.html")

def sign_in(request):
    return render(request,"sign_in.html")


def sign_up(request):
    return render(request,"sign_up.html")



def live_prediction_view(request):
    return render(request, "test.html")
