// to compile: mpicc jacobi-mpi.c -o jacobi-mpi -fopenmp
// to run: mpirun -np 1 ./jacobi-mpi N P T
// ONDE N = ORDEM DA MATRIZ, P = NUMERO DE PROCESSOS e T = NUMERO DE THREADS POR PROCESSO.
//
// Para executar em vários nodes do cluster, rodar, por exemplo:
//  mpirun -np 1 --hostfile halley.txt 01-spawn-simple
// a funcao MPI_Comm_spawn() pode indicar o hostfile tambem.
//
//
// A aplicação funciona com -np maiores que 1. Permite exemplificar que se trata de uma primitiva coletiva 
// e que apenas um processo cria os processos filhos.
// to run: mpirun -np 8 01-spawn-simple
//
// o numero de processos a gerar é limitado pelo MPI em função do nr de slots.
// Os slots determinam o nr de processos a escalonar em cada processador
// o padrão para o slot é um processo por core, mas pode mudar.
//
// --map-by node faz um round-robin por node e nao por core do node
//
// --use-hwthread-cpus if you want Open MPI to default to the number of hardware threads instead of the 
//           number of processor cores
// 
// --oversubscribe  to ignore the number of available slots when deciding the number of processes to launch.
// 
// -host namehost:slots     to determine the number of slots
//
// -host namehost1:slots,namehost2:slots,namehost3:slots (ele trava se não colocar o hostfile no spawn)
//
// 
// para o MPI_Comm_spawn executar sobre diferentes nós, deve-se executar
// o mpirun com --hostfile ou --host
// isso permite registrar mais slots à execução que serão utilizados no spawn
// veja manpage do mpirun para aplicar corretamente os argumentos 
// https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php
// veja https://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-2.0/node95.htm#Node95
// 
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// o numero de processos a gerar é limitado pelo MPI em função 
// de fatores como, por exemplo, o número de nucleos nos processadores
// neste caso só um processador é usado).

int main( int argc, char *argv[])
{
    int P = atoi(argv[2]); //Usado somente pelo processo pai 'criador' para um check inicial.
    int N = atoi(argv[1]); //Usado somente pelo processo pai 'criador' para um check inicial.
    int my_rank;
    int errcodes[P];
    
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
	
    MPI_Comm parentcomm, intercomm;

    MPI_Init( &argc, &argv );
    MPI_Comm_get_parent( &parentcomm );
    MPI_Get_processor_name(processor_name, &name_len);
  
    
    if (parentcomm == MPI_COMM_NULL) //Processo pai executa esse bloco e spawna os filhos que executarão o algoritmo.
    {
	    int root = 0;
	    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        if(P<=N){
            MPI_Comm_spawn( "./jacobi-mpi", argv, P, MPI_INFO_NULL, root, MPI_COMM_WORLD, &intercomm, errcodes );
        }else{
            MPI_Comm_spawn( "./jacobi-mpi", argv, N, MPI_INFO_NULL, root, MPI_COMM_WORLD, &intercomm, errcodes );
        }
        printf("I'm the parent number %d on processor %s.\n", my_rank, processor_name);
		fflush(0);
    }
    else //Bloco que será executado pelos P filhos invocados.
    {
      MPI_Comm_rank(parentcomm, &my_rank);
      int quantoPercorrer;       //Será usada no controle do percorrimento do vetor variavel de cada processo.
      int lineCriteria = 1;     //Flag usada na determinacao de convergencia.
      int reducaoCriterio = 1;  //Variavel que conterá o valor da reducao MPI do criterio de convergencia.
      int orderOfMatrix = atoi(argv[2]);      //Parametro de entrada, ordem da matriz gerada.
      int NumberOfProcecess = atoi(argv[3]); //Parametro de entrada, numero de processos gerados.
      int numberOfThreads = atoi(argv[4]);  //Parametro de entrada, numero de threads por processo.
      int i,j,k=0;      //Variaveis auxiliares de controle em loops.
      int numIter=0; // Contador do numero de iteracoes.
      //Definindo a proporcao de quantas linhas serao enviadas pra cada processo.  N/P
      int proporcao = orderOfMatrix/NumberOfProcecess;
      if(proporcao == 0){ // Se o numero do processos for maior que a ordem da matriz passaremos 1 linha pra cada e haverao processos ociosos.
        proporcao=1;
        NumberOfProcecess=orderOfMatrix; // Caso haja mais processos que linhas na matriz, define o maximo de processos assim.
      }
      // IMPORTANTE: Politica de distribuicao de tarefas -> Coloca o resto da divisao no ultimo processo.
      // Caso a conta nao seja exata, alocaremos o resto das linhas que sobram no processo de rank mais alto.
      int quantidadePiorCaso = proporcao*orderOfMatrix + (orderOfMatrix%NumberOfProcecess)*orderOfMatrix; //Utilizado para determinar quantos elementos da matriz serão enviados no ultimo processo.
      int quantidadePiorCasoB = proporcao + (orderOfMatrix%NumberOfProcecess); // Utilizado para determinar quantos elementos do vetor B serão enviados ao ultimo processo.

      int *vetor_env, *vetor_recB, *vetor_rec, *vetorQuantidades, *vetorDisplacements, *vetorQuantidadesB, *vetorDisplacementsB; //Ponteiros que serão usados no controle da comunicação MPI.
      int **matrixAux; //Matriz auxiliar que vai facilitar a manipulaçao dos dados em cada processo.
      // Alocacao dinamica dos vetores.
      vetor_env=(int*)malloc(orderOfMatrix*sizeof(int));
      vetor_rec=(int*)malloc(quantidadePiorCaso*sizeof(int));
	  vetor_recB=(int*)malloc(quantidadePiorCasoB*sizeof(int));
      vetorQuantidades=(int*)malloc(NumberOfProcecess*sizeof(int));
      vetorDisplacements=(int*)malloc(NumberOfProcecess*sizeof(int));
      vetorQuantidadesB=(int*)malloc(NumberOfProcecess*sizeof(int));
      vetorDisplacementsB=(int*)malloc(NumberOfProcecess*sizeof(int));
      int matrix[3][3];
      int bVector[4];
      if(my_rank == 0){ //Processo 0 é o responsavel por gerar a matriz aleatória, separar e enviar as tarefas pros outros. Como um "gerente".
        printf("N: %d, P: %d, T: %d \n",orderOfMatrix, NumberOfProcecess, numberOfThreads);
        //INICIO GERACAO DA MATRIZ E VETOR B;
        matrix[0][0] =4; //Diagonal Princ (geracao hardcoded pra testes.)
		matrix[0][1] =2;
		matrix[0][2] =1;
		// matrix[0][3] =0;
        matrix[1][0] =1;
        matrix[1][1] =3; //Diagonal Princ
        matrix[1][2] =1;
       // matrix[1][3] =0;
        matrix[2][0] =2;
        matrix[2][1] =3;
        matrix[2][2] =6; //Diagonal Princ
       // matrix[2][3] =0;
       // matrix[3][0] =5;
       // matrix[3][1] =5;
        //matrix[3][2] =5;
        //matrix[3][3] =16; //Diagonal Princ

        //int bVector[4] = {7, -8, 6, 5};
        bVector[0]=7;
        bVector[1]=-8;
        bVector[2]=6;
        bVector[3]=5;
		vetor_env = bVector;
        // FIM GERACAO MATRIZ E VETOR B
        if(orderOfMatrix%NumberOfProcecess !=0){ // Definição das quantidades e displacements para o scatter variavel.
          #pragma omp parallel for private(i) num_threads(numberOfThreads)
          for(i=0;i<NumberOfProcecess-1;i++)
          {
          vetorQuantidades[i] = proporcao*orderOfMatrix;
          vetorDisplacements[i] = proporcao*orderOfMatrix*i;
          vetorQuantidadesB[i] = proporcao;
          vetorDisplacementsB[i]=proporcao*i;
          }
          vetorQuantidades[NumberOfProcecess-1] = quantidadePiorCaso;
          vetorDisplacements[NumberOfProcecess-1] = proporcao*orderOfMatrix*(NumberOfProcecess-1);
          vetorQuantidadesB[NumberOfProcecess-1] = quantidadePiorCasoB;
          vetorDisplacementsB[NumberOfProcecess-1]= proporcao*(NumberOfProcecess-1);
        }else{ // Caso a conta seja exata, fica mais simples e todos recebem a mesma fatia:
          #pragma omp parallel for private(i) num_threads(numberOfThreads)
          for(i=0;i<NumberOfProcecess;i++)
          {
          vetorQuantidades[i] = proporcao*orderOfMatrix;
          vetorDisplacements[i] = proporcao*orderOfMatrix*i;
          vetorQuantidadesB[i] = proporcao;
          vetorDisplacementsB[i]=proporcao*i;
          }
        }
      }else{ //Processos que não são o 0 só dizem que foram invocados corretamente.
        printf("I'm the spawned process number %d on processor %s.\n", my_rank, processor_name);
      }
      // Comunicacao coletiva, envio das linhas da matriz "A" e dos elementos do vetor "B" para os seus respectivos processos.
      MPI_Scatterv(matrix[0], vetorQuantidades, vetorDisplacements, MPI_INT, vetor_rec, quantidadePiorCaso, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Scatterv(vetor_env, vetorQuantidadesB, vetorDisplacementsB, MPI_INT, vetor_recB, quantidadePiorCasoB, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      // Fim da comunicação coletiva.
      if(my_rank < orderOfMatrix){ // Condição importante para evitar SEGFAULT, somente processos com dados alocados devem executar.
        // Prints para averiguar o funcionamento correto. (DEBUG)
        printf("\nThere are %d processes. I am process %d from processor %s.\n", NumberOfProcecess, my_rank, processor_name);
        printf("Received Values \n");
        if(my_rank == NumberOfProcecess-1){
          for(i=0;i<quantidadePiorCaso;i++){
            printf("%d ",vetor_rec[i]);
          }
          printf("\n BValues: ");
          for(i=0;i<quantidadePiorCasoB;i++){
            printf("%d ",vetor_recB[i]);
          }
          printf("\n");
        }else{
          for(i=0;i<orderOfMatrix*proporcao;i++)
            printf("%d ",vetor_rec[i]);
          printf("\n BValues: ");
          for(i=0;i<proporcao;i++){
            printf("%d ",vetor_recB[i]);
          }
          printf("\n");
        }
        // FIM DOS PRINTS DE DEBUG
        // PODEMOS INICIAR AGORA O CRITERIO DE CONVERGENCIA ***************************************************
        printf("\nInicio criterio convergencia\n");
        //Alocacao de matriz auxiliar para facilitar manipulacao dos dados. sempre temos que tratar o pior caso e o caso geral...
        if(my_rank == NumberOfProcecess-1){
          matrixAux = malloc((proporcao+(orderOfMatrix%NumberOfProcecess))*sizeof(int));
          for(i=0;i<orderOfMatrix;i++){
            matrixAux[i]=malloc(orderOfMatrix*sizeof(*matrixAux[i]));
          }
        }else{
          matrixAux = malloc(proporcao*sizeof(int));
          for(i=0;i<orderOfMatrix;i++)
              matrixAux[i]=malloc(orderOfMatrix*sizeof(*matrixAux[i]));
        }
        //Preenchimento da matriz auxiliar.
        k=0;
        //PULO DO GATO - CONTROLE DE FLUXO CASO SEJA O ULTIMO PROCESSO.
        quantoPercorrer = (my_rank == (NumberOfProcecess-1)) ? quantidadePiorCasoB : proporcao;
        printf("Matriz-Auxiliar\n");
        if(my_rank<orderOfMatrix){
          for(i=0;i<quantoPercorrer;i++)
          {
            for(j=0;j<orderOfMatrix;j++){
              matrixAux[i][j] = vetor_rec[k];
              k++;
              printf("%d ", matrixAux[i][j]);
            }
          printf("\n");
          }
        }
      }
    MPI_Barrier(MPI_COMM_WORLD);
    //Agora todos processos estão prontos para iniciar o teste da convergencia.
    //INICIO DO CRITERIO PROPRIAMENTE DITO
    int* lineSum = malloc(quantidadePiorCasoB*sizeof(long int)); //Vetor que vai ser usado no calculo da convergencia.
    if(my_rank < orderOfMatrix){ //Preenchimento com 0
        #pragma omp parallel for private(i) num_threads(numberOfThreads) 
        for (i = 0; i < orderOfMatrix; i++)
        { 
      		lineSum[i] = 0;
   		}
        int w = my_rank*proporcao; //controla em qual linha da matriz original estamos.
        if(my_rank<orderOfMatrix){ //Somente processos com dados validos executam
          #pragma omp parallel for private(i,j) num_threads(numberOfThreads)
          for(i=0;i<quantoPercorrer;i++) 
          {
            if(lineCriteria){
              for(j=0;j<orderOfMatrix;j++){
                if(j != w+i)
                  lineSum[i] = lineSum[i] + matrixAux[i][j];
                if(lineSum[i] > matrixAux[i][w+i]){
                  lineCriteria=0;
                  printf("\n Falhou no critério, na linha %d\n", w+i+1);
                  break;
                }
              }
            }
          }
        }
    }
    // Nesse momento possuimos o resultado do criterio para cada linha, podemos fazer a reducao e caso algum falhe o criterio falha.
    MPI_Reduce(&lineCriteria, &reducaoCriterio, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reducaoCriterio, 1, MPI_INT, 0, MPI_COMM_WORLD); //Envia a todos os processos se o criterio foi bem sucedido, se sim continua,
                                                                 // senao os processos encerram e o programa termina.
    if(reducaoCriterio){
        printf("Passou pelo criterio das linhas.\n");
    }else{
        fflush(stdout);
        MPI_Finalize();
        return 0;
    }
    // Chegando nesse ponto sabemos que a matriz converge, agora devemos realizar o processo iterativo de jacobi
    //e assim determinar uma solucao numerica pro sistema linear.
    float *lastResults, *currentResults, *resultados; // Variaveis que serao usadas no criterio.
    int *quantiaResultados, *displacementResultados;
    resultados = (float*)malloc(orderOfMatrix*sizeof(float)); //o vetor de resultados contem a aproximacao numerica mais recente.
    if(my_rank == 0){ //O processo rank 0 gerente atribui os valores adequados pros vetores do GATHERV MPI
        quantiaResultados= (int*)malloc(NumberOfProcecess*sizeof(int));
        displacementResultados = (int*)malloc(NumberOfProcecess*sizeof(int));
        for(i=0;i<NumberOfProcecess;i++){
          if(i==(NumberOfProcecess-1)){ //Ultimo processo
            quantiaResultados[i]=quantidadePiorCasoB;
            displacementResultados[i]=proporcao*i;
          }else{ //Outros processos
            quantiaResultados[i]=proporcao;
            displacementResultados[i]=proporcao*i; // PULO DO GATO multiplicar por i.
          }
        }
        //PRINTS DE DEBUG
        printf("\nVetor quantia de resultados\n");
        for(i=0;i<NumberOfProcecess;i++){
            printf("%d ",quantiaResultados[i]);
        }
        printf("\nVetor displacements de resultados\n");
        for(i=0;i<NumberOfProcecess;i++){
            printf("%d ",displacementResultados[i]);
        }
        // FIM PRINTS DE DEBUG
    }
    //INICIO DAS ITERACOES
    //DEFINICAO E PREENCHIMENTO DO VETOR DE ULTIMOS RESULTADOS POR CADA PROCESSO. (São os valores iniciais no caso)
    if(my_rank<orderOfMatrix){
        lastResults = (float*)malloc((quantidadePiorCasoB)*sizeof(float));
        currentResults = (float*)malloc((quantidadePiorCasoB)*sizeof(float));
        int w = my_rank*proporcao;
        printf("\nPreenchimento vetor last results do rank: %d\n",my_rank);
        #pragma omp parallel for private(i) num_threads(numberOfThreads)
        for(i=0;i<quantoPercorrer;i++)
        {
          lastResults[i] = (float)vetor_recB[i]/(float)matrixAux[i][w+i];
          printf("%d / %d \n",vetor_recB[i], matrixAux[i][w+i]);
        }
    }
    // Vetor que será enviado é o lastResults, com X elementos no pior caso, do tipo float, o numero de elementos recebido de cada processo esta no vetor de contagens
    //o displacement de cada elemento está no vetor displacement o tipo de dado é float, a raiz(recebedor) é o processo 0 e o comunicador é o do grupo)
    if(my_rank==NumberOfProcecess-1){
        MPI_Gatherv(lastResults, quantidadePiorCasoB, MPI_FLOAT, resultados, quantiaResultados, displacementResultados, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gatherv(lastResults, proporcao, MPI_FLOAT, resultados, quantiaResultados, displacementResultados, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    //DEBUG PARA CHECAR O RECEBIMENTO CORRETO PELO GATHERV
    if(my_rank==0){
        printf("\nPrint do vetor recebido via gather:\n");
        for(i=0;i<orderOfMatrix;i++){
          printf("%.3f ", resultados[i]);
        }
        printf("Inicio das iterações.\n");
    }
    //DEBUG PARA CHECAR O RECEBIMENTO CORRETO PELO GATHERV
    //Broadcast do vetor inicial para ser usado pelos processos e iterado.
    MPI_Bcast(resultados, orderOfMatrix, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // INICIO DAS ITERACOES DO METODO!!!!
    int w;
    float maximoDiff, maximoValor;
    float redMaxVal, redMaxDiff;
    do { // Loop que dura até a condição de convergencia ser atingida.
        if(my_rank<orderOfMatrix){
          printf("\nInicio de uma iteracao\n");
          maximoValor=0;
          maximoDiff=0;
          redMaxDiff=0;
          redMaxVal=0;
          int thread_num, nthreads;
          w = my_rank*proporcao;
          #pragma omp parallel for private(j) reduction(max:maximoValor) reduction(max:maximoDiff) num_threads(numberOfThreads)
          for(i=0;i<quantoPercorrer;i++)
          {
            printf("\n Line %d: ", w+i);
            currentResults[i] = 0;
            for(j=0; j<orderOfMatrix; j++)
            {
              //DEBUG OMP   
              #pragma omp critical
              {
              thread_num = omp_get_thread_num();    
              nthreads = omp_get_num_threads( );
              printf(" \nHello-world da thread %d na região paralela, Num_threads aqui: %d, i: %d, j: %d\n", thread_num, nthreads,i,j);           
              } 
              //DEBUG OMP 
              if(j == w+i){
                currentResults[i] += ((float)vetor_recB[i]/(float)matrixAux[i][w+i]);
                printf("\n + %d / %d", vetor_recB[i],matrixAux[i][w+i]);
              }
              else{
                currentResults[i] -=  ((float) matrixAux[i][j] * resultados[j] / (float) matrixAux[i][w+i]);
                printf("\n - (%d * %.2f) / %d", vetor_recB[i], resultados[j], matrixAux[i][w+i]);
              }
            }
            printf("\n");
            printf("%f ",currentResults[i]);

            if(fabs(currentResults[i]) > maximoValor){
              maximoValor = fabs(currentResults[i]);
              printf("\nmaximoValorAtualizado: %f",currentResults[i]);
            }
            if(fabs(currentResults[i]-resultados[w+i])> maximoDiff){
              maximoDiff = fabs(currentResults[i]-resultados[w+i]);
              printf("\nmaximoDiffAtualizado: (%.4f - %.4f) = %.4f",currentResults[i],resultados[w+i],maximoDiff);
            }
          }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      //Reducoes MPI dos valores de maximo usados no calculo do criterio de parada.
      MPI_Allreduce(&maximoValor, &redMaxVal, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&maximoDiff, &redMaxDiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
      printf("\nMaximoVal recebido pos reducao: %f", redMaxVal);
      printf("\nMaximoDiff recebido pos reducao: %f", redMaxDiff);
      printf("\nMaxDiff/MaxVal:  %f \n", redMaxDiff/redMaxVal);
      //Gather dos resultados da iteracao mais recente e depois broadcast para todos processos possuirem o vetor mais atualizado.
      if(my_rank==NumberOfProcecess-1){
        MPI_Gatherv(currentResults, quantidadePiorCasoB, MPI_FLOAT, resultados, quantiaResultados, displacementResultados, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }else{
        MPI_Gatherv(currentResults, proporcao, MPI_FLOAT, resultados, quantiaResultados, displacementResultados, MPI_FLOAT, 0, MPI_COMM_WORLD);
      } 
      MPI_Bcast(resultados, orderOfMatrix, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      if(my_rank==0){ //Contador de iteracoes.
        numIter++;
      }
      printf("\nResultado recebido pos iteracao\n");
      for(i=0; i<orderOfMatrix; i++){
        printf("%f ", resultados[i]);
      }
    }while(redMaxDiff/redMaxVal>=0.0015);
    // TESTE FINAL DE ESCOLHA DA LINHA DA MATRIZ
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == 0){
        int linhaEscolhida=1;
        float resultadoFinal=0;
        //printf("Escolha a linha da matriz para testar o resultado obtido: ");
        //scanf("%d", &linhaEscolhida);
        for(i=0;i<orderOfMatrix;i++){
          resultadoFinal += matrix[linhaEscolhida][i]*resultados[i];
        }
        printf("\nTemos: %.4f = %d",resultadoFinal,vetor_env[linhaEscolhida]);
        printf("\nConvergiu em %d iteracoes",numIter);
    }
    ///
    // DEPOIS DAQUI ENGLOBA O PROCESSO ROOT TBM  
    } // FIM DO ELSE QUE SEPARA OS SPAWNS DO ROOT PROCESS
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
