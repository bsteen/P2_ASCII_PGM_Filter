#define GLEW_STATIC
#include <GL/glew.h>
#include <SFML/Window.hpp>
#include <iostream>
#include "Window.hpp"
using namespace std;

const GLchar* vertexSource =
            "#version 150 core\n"
            "in vec2 position;"//0
            "in vec2 texcoord;"//sizeof(float) * 2

            "out vec2 Texcoord;"
            "void main() {"
                "Texcoord = texcoord;"
                "gl_Position = vec4(position, 0.0, 1.0);"
            "}";

const GLchar* fragmentSource =
            "#version 150 core\n"
            "in vec2 Texcoord;"

            "out vec4 outColor;"
            "uniform sampler2D tex;"
            "void main(){"
                "outColor = texture(tex, Texcoord);"
            "}";

void printShaderCompileStatus(GLuint shader, string name){
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

    if(!status){
        cout <<  name + " shader did not compile!" << endl;
    }
}

void displayImage(int* imageData, int width, int height){
    //Initiliaze window setting values
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    //Create window
    sf::Window window(sf::VideoMode(width, height), "Image Output", sf::Style::Close, settings);

    window.setFramerateLimit(60);

    glewExperimental = GL_TRUE;//Force GLEW to use a modern OpenGL method for checking if a function is available.
    glewInit();

    GLuint vertexArrayObject;//Create a vertex array object (VOA). It stores information everytime glVertexAttribPointer is called.
    glGenVertexArrays(1, &vertexArrayObject);
    glBindVertexArray(vertexArrayObject);//When switching to different vertex data and vertex formats, just need to bind the right VOA.

    GLfloat vertices[] = {
        //Position      Texcoords
        -1.0f,  1.0f, 0.0f, 0.0f, // Top-left
        1.0f,  1.0f, 1.0f, 0.0f, // Top-right
        1.0f, -1.0f, 1.0f, 1.0f, // Bottom-right
        -1.0f, -1.0f, 0.0f, 1.0f  // Bottom-left
    };//OpengGL has origin at center of screen; Textures have origin at top left.

    GLuint vertexBufferObject;
    glGenBuffers(1, &vertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);//Make Vertex Buffer Object the active array buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);//Copy vertices to the active buffer (vertexBufferObject).
    //GL_STATIC_DRAW => The vertex data will be uploaded once and drawn many times.

    GLuint elementArrayObject;//I don't know the purpose of this.
    glGenBuffers(1, &elementArrayObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementArrayObject);
    GLuint elements[] = {
      0, 1, 2,
      2, 3, 0
    };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    //Compile vertex shader;
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);//Create vertex shader and load data into it
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);//Compile the vertex shader now
    printShaderCompileStatus(vertexShader, "Vertex");//Check if it compiled.

    //Load fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);//Create fragment shader and load data into it
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);//Compile the fragment shader now
    printShaderCompileStatus(fragmentShader, "Fragment");

    //Link vertex and fragment shader into a program.
    GLuint shaderProgram = glCreateProgram();//Create program out of the two shaders.
    glAttachShader(shaderProgram, vertexShader);//Only one vertex buffer can be active at a time.
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");//Fragements use outColor, which is the texture values.
    glLinkProgram(shaderProgram);//After attaching the shaders, they must be linked toegther.
    glUseProgram(shaderProgram);//Start using the program. Only one program can be active at a time.

    GLint vertexByteSize = sizeof(GLfloat) * 4;//vec2 position and vec2 texcoord.

    //Specify the layout of the vertex data
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");//Need to define how attributes are formatted and ordered in the vertex shader.
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, vertexByteSize, 0);//Info is now also stored in the vertexArrayObject created at the start.

    GLint texAttrib = glGetAttribLocation(shaderProgram, "texcoord");
    glEnableVertexAttribArray(texAttrib);
    glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, vertexByteSize, (void*)(sizeof(GLfloat) * 2));

    unsigned char newImageData[width * height];//For some reason, OpenGL doesn't seem to like the image data as ints,
    for(int i = 0; i < width * height; i++){     //so I convert it to unsigned chars before I make it a texture.
        newImageData[i] = imageData[i];
    }

    //Create a texture from imageData
    GLuint imageTexture;
    glGenTextures(1, &imageTexture);
    glBindTexture(GL_TEXTURE_2D, imageTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, newImageData);//Make the texture.

    //Set texture parameters.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);//X wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);//Y wrapping
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);//Filtering types for when the image is rescaled.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //Run main window loop
    bool running = true;
    while (running){
        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)){
            switch (windowEvent.type){
                case sf::Event::Closed:
                    running = false;
                    break;
                case sf::Event::KeyPressed:
                    if (windowEvent.key.code == sf::Keyboard::Escape){
                        running = false;
                    }
                    break;
            }
        }

        //Display image
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        window.display();
    }

    //Clean up
    glDeleteTextures(1, &imageTexture);
    glDeleteProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteBuffers(1, &elementArrayObject);
    glDeleteBuffers(1, &vertexBufferObject);
    glDeleteVertexArrays(1, &vertexArrayObject);
}
