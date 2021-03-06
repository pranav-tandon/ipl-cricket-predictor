from django.shortcuts import render,HttpResponse
from cricketAI.predictions.pred import *


from .forms import *

def main(request):
    return render(request, 'cricketAI/main.html', {})

def gallery(request):
        return render(request, 'cricketAI/gallery.html', {})

def about(request):
        return render(request, 'cricketAI/about.html', {})

def first_inn(request):
    if request.method == 'POST':
        title_form = InningsFirst(request.POST)
        if title_form.is_valid():
            team1 = title_form.cleaned_data['team1']
            team2 = title_form.cleaned_data['team2']
            venue = title_form.cleaned_data['venue']
            overs_played = title_form.cleaned_data['overs_played']
            runs = title_form.cleaned_data['runs']
            wickets = title_form.cleaned_data['wickets_fallen']
            run_predicted = int(predict_1st_inn(team1,team2,runs,6*overs_played,wickets,venue)) + 1
            return render(request, 'cricketAI/firstinn.html', context={'form1': title_form,"runs":run_predicted})

    else:
        title_form = InningsFirst()

    return render(request, 'cricketAI/firstinn.html', context={'form1': title_form})


def second_inn(request):
    if request.method == 'POST':
        title_form = InningsSecond(request.POST)
        if title_form.is_valid():
            team1 = title_form.cleaned_data['team1']
            team2 = title_form.cleaned_data['team2']
            venue = title_form.cleaned_data['venue']
            runs = title_form.cleaned_data['runs']
            overs_played = title_form.cleaned_data['overs_played']
            target_chasing = title_form.cleaned_data['target_set']
            wickets = title_form.cleaned_data['wickets_fallen']

            result = predict_2nd_inn(team1,team2,runs,6*overs_played,wickets,target_chasing,venue)
            by_run = 0
            by_wicket = 0
            if result[0]>0.5:
                winner = get_team(team1)
                probab = result[0] * 100
                by_wicket = int(result[1])
            elif result[0]<0.5:
                winner = get_team(team2)
                probab = (1-result[0])*100
                by_run = int(result[1])
            else:
                winner = "Can be any one"
                probab = 50

            if(result[2]>=120):
                result[2]=119
            end = str(int(result[2]/6)+1)

            return render(request, 'cricketAI/secondinn.html', context={'form2': title_form,
                            "winner":winner,"probab":probab,"by_run":by_run,"by_wicket":by_wicket,"end":end})

    else:
        title_form = InningsSecond()

    return render(request, 'cricketAI/secondinn.html', context={'form2': title_form})



def prematch(request):
    if request.method == 'POST':
        title_form = PreMatch(request.POST)
        if title_form.is_valid():
            team1 = title_form.cleaned_data['team1']
            team2 = title_form.cleaned_data['team2']
            venue = title_form.cleaned_data['venue']
            probab = pre_match_predict("2016",team1,team2,venue)
            if probab > 0.5 :
                winner = get_team(team1)
                probab = probab * 100
            else:
                winner =  get_team(team2)
                probab = (1- probab) * 100

            return render(request, 'cricketAI/pre_pred.html', context={'form3': title_form,"winner":winner,"probab":probab})

    else:
        title_form = PreMatch()

    return render(request, 'cricketAI/pre_pred.html', context={'form3': title_form})
