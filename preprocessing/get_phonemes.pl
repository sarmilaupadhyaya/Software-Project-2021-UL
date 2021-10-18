#!/usr/bin/perl
use strict;
use warnings;

use HTTP::Request;                      #to encode an HTTP request
use LWP::UserAgent;                     #class for web user agent (to send request and receive response)
use Data::Dumper;

my $ua = LWP::UserAgent->new;

if($#ARGV != 3)
{
        print "eLiteHTS_client.pl file(.txt or .textgrid) resource_type(texts or textgrids) output_format (hts, dls, textgrid_hts) mode (train or run)\n";
        exit(1);
}

my $server_uri_base = 'http://cental.uclouvain.be/elitehts/v1/';
#----------------------------------------------------------------------
#-- Read content file
#----------------------------------------------------------------------

print "Read content...\n";
my $input_file = $ARGV[0];
open(IN, $input_file);
binmode(IN, ':encoding(latin1)');

my $input_content = "";
while(my $line =<IN>)
{
        #print $line."\n";
        $input_content .= $line;
}

$input_content .= "\n";


#----------------------------------------------------------------------
#-- OPTIONS of /resource
#----------------------------------------------------------------------
print "Get resource options...\n";


my $res_type = $ARGV[1];

my $uri = $server_uri_base.$res_type;
print "URI ".$uri."\n";
my $req = HTTP::Request->new( 'OPTIONS' => $uri);
$req->header( 'Accept'       => 'application/xml' );
my $res = $ua->request($req);
if($res->code eq '204')
{
        print "Allow :". $res->header('Allow')."\n";
}


#----------------------------------------------------------------------
#-- POST the resource
#----------------------------------------------------------------------
print "Post a new resource...\n";

$uri = $server_uri_base.$res_type;
print "URI ".$uri."\n";
$req = HTTP::Request->new( 'POST' => $uri);
$req->header( 'Content-Type' => 'text/plain;charset=iso-8859-1' );
$req->header( 'Accept'       => 'application/xml' );
$req->content($input_content);

#-- Launch the request
my $res1 = $ua->request($req);

#-- If creation is not successful
if ( $res1->code ne '201' ) {
        print $res1->status_line."\n";
        print $res1->decoded_content."\n";
        exit(1);
}
else
{
        print $res1->decoded_content."\n";
}

#----------------------------------------------------------------------
#-- GET options of /resource/:id/
#----------------------------------------------------------------------
print "GET options of /resource/:id/...\n";

#-- Create a request and set the headers
$uri = $res1->header('Location');
print "URI POST ".$uri."\n";
$req = HTTP::Request->new( 'OPTIONS' => $uri);
$req->header( 'Accept'       => 'application/xml' );
my $res2 = $ua->request($req);
if($res2->code eq '204')
{
        print "Allow :". $res2->header('Allow')."\n";
}

#----------------------------------------------------------------------
#-- GET ouput content (hts, dls or textgrid_hts format)
#----------------------------------------------------------------------
print "GET ouput content...\n";
my $output_format = $ARGV[2];
my $mode = $ARGV[3];

#-- get resource location
$uri = $res1->header('Location');
$uri .= "/".$output_format."?mode=".$mode;
print "URI ".$uri."\n";
#-- Create a request and set the headers
$req = HTTP::Request->new('GET' => $uri);
$req->header('Accept' => 'text/plain');

#-- Launch the request
my $res3 = $ua->request($req);


#-- If retrieve is not successful
if ( $res3->code ne '200' ) {
        print $res3->status_line."\n";
        print $res3->decoded_content."\n";
        print exit(1);
}

#-- If retrieve is successful
print $res3->decoded_content;

#----------------------------------------------------------------------
#-- Delete ressource
#----------------------------------------------------------------------
print "DELETE resource...\n";

#-- get resource location
$uri = $res1->header('Location');
print "URI ".$uri."\n";

#-- Create a request and set the headers
$req = HTTP::Request->new('DELETE' => $uri);
$req->header('Accept' => 'application/xml');

#-- Launch the request
my $res4 = $ua->request($req);
print $res4->code."\n";
print $res4->decoded_content."\n";
